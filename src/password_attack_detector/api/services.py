"""Serving orchestration: compose the frozen system, decide nothing new.

The whole request path is one composition of parts that already exist::

    request
      -> canonical AuthEvent            (data.schemas)
      -> point-in-time feature rows     (features.engine.FeatureEngine)
      -> rule verdicts                  (detection.engine.DetectionEngine)
      -> event risk                     (detection.scoring.RiskScorer)
      -> frozen model decision          (ml.predictions.predict_binary)
      -> frozen fusion                  (ml.fusion.fuse)
      -> response

Nothing in this module computes a feature, re-derives a threshold, re-weights a
rule, or blends the two systems' scores.  Where a quantity is needed, it is
obtained by calling the frozen implementation that owns it, and where a frozen
implementation refuses, the refusal is reported rather than routed around.

**Startup is fail-closed and fail-visible.**  :func:`build_runtime` never raises:
it returns a runtime whose components each carry their own state and, when not
ready, a stable reason code.  A component that failed leaves readiness false and
detection refused; it never leaves a silently substituted alternative in place.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from password_attack_detector import __version__
from password_attack_detector.api.config import APISettings
from password_attack_detector.api.errors import APIError, ErrorCode, sanitize
from password_attack_detector.api.schemas import (
    AnchorDetection,
    BatchDetectionResponse,
    ComponentReport,
    ComponentState,
    DetectionResponse,
    HybridLayerResult,
    MLLayerResult,
    ModelInfoResponse,
    ReadinessResponse,
    ReadinessState,
    RuleCatalogResponse,
    RuleLayerResult,
    RuleSummary,
    SystemStatusResponse,
    VersionResponse,
    WindowRequestBase,
    WindowSummary,
)
from password_attack_detector.config import load_settings
from password_attack_detector.data.privacy import PseudonymService
from password_attack_detector.data.schemas import AuthEvent
from password_attack_detector.deployment.bundle import (
    ServingBundle,
    load_serving_bundle,
)
from password_attack_detector.detection.catalog import RULE_CATALOG, RuleCatalog
from password_attack_detector.detection.config import (
    DetectionConfig,
    load_detection_config,
)
from password_attack_detector.detection.engine import DetectionEngine
from password_attack_detector.detection.schemas import (
    DETECTION_SCHEMA_VERSION,
    RiskAssessment,
)
from password_attack_detector.detection.scoring import SCORING_VERSION, RiskScorer
from password_attack_detector.exceptions import (
    ArtifactNotFoundError,
    PasswordAttackDetectorError,
    PseudonymizationError,
)
from password_attack_detector.features.catalog import (
    ANCHOR_EVENT_ID,
    ANCHOR_EVENT_TIME,
    FeatureCatalog,
    build_catalog,
)
from password_attack_detector.features.config import (
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
    load_feature_config,
)
from password_attack_detector.features.engine import FeatureEngine
from password_attack_detector.logging_config import get_logger
from password_attack_detector.ml.config import MLConfig, load_ml_config
from password_attack_detector.ml.dataset import assemble_serving_batch
from password_attack_detector.ml.enums import (
    FusionStrategy,
    ServingScope,
    is_probability,
)
from password_attack_detector.ml.features import (
    EligibleFeatureList,
    load_feature_allowlist,
    resolve_eligible_features,
)
from password_attack_detector.ml.fusion import (
    FUSION_SCHEMA_VERSION,
    MLEvidence,
    RuleEvidence,
    StackedFusionState,
    fuse,
)
from password_attack_detector.ml.ledger import ExperimentLedger, TestEvaluationRecord
from password_attack_detector.ml.predictions import (
    BinaryPrediction,
    FrozenChampion,
    predict_serving_binary,
    verify_inference_feature_contract,
)
from password_attack_detector.ml.schemas import ML_SCHEMA_VERSION
from password_attack_detector.ml.test_evaluation import (
    EVALUATION_RECEIPT_FILE,
    EVALUATIONS_DIR,
)

__all__ = [
    "COMPONENT_FEATURE_CONTRACT",
    "COMPONENT_FUSION",
    "COMPONENT_ML_CHAMPION",
    "COMPONENT_MODEL_ARTIFACTS",
    "COMPONENT_RULE_ENGINE",
    "SERVING_SCOPE",
    "FusionRuntime",
    "MLRuntime",
    "RuleRuntime",
    "RuntimeState",
    "build_runtime",
    "detect_batch",
    "detect_single",
    "hybrid_layer",
    "model_info_document",
    "readiness_document",
    "rule_catalog_document",
    "system_status_document",
    "version_document",
]

_log = get_logger(__name__)

COMPONENT_RULE_ENGINE: Final[str] = "rule_engine"
COMPONENT_FEATURE_CONTRACT: Final[str] = "feature_contract"
COMPONENT_MODEL_ARTIFACTS: Final[str] = "model_artifacts"
COMPONENT_ML_CHAMPION: Final[str] = "ml_champion"
COMPONENT_FUSION: Final[str] = "fusion"

#: What a live request is, in the ML layer's own vocabulary.
#:
#: Not a split.  ``ServingScope`` is a separate type from ``MLSplit`` with a
#: separate vocabulary, so a request cannot be filed as TRAIN, validation, TEST,
#: or the novel-anomaly holdout -- not by configuration, not by a parameter, and
#: not by an accidental default.  No split table is read, nothing is published,
#: no label is joined, and no evaluation artifact, receipt, or metric is written.
#:
#: One member, and it must stay a constant rather than becoming a *choice*:
#: serving rows are all in one scope, always, so a caller cannot select a
#: different scoring population.
SERVING_SCOPE: Final[ServingScope] = ServingScope.LIVE


# ---------------------------------------------------------------------------
# Immutable runtime state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuleRuntime:
    """The prepared rule layer.

    The engine and the scorer are both immutable once constructed: the engine's
    rules are prepared in ``__init__`` and its ``run`` builds all per-call state
    locally, and the scorer holds only its configuration.  Both are therefore
    shared across requests rather than rebuilt per request.
    """

    config: DetectionConfig
    engine: DetectionEngine
    scorer: RiskScorer
    #: The catalog the engine was prepared against.  Carried rather than assumed
    #: so the published rule document describes the rules that actually ran.
    catalog: RuleCatalog


@dataclass(frozen=True, slots=True)
class MLRuntime:
    """The verified frozen champion and the feature contract it was fitted under."""

    champion: FrozenChampion
    eligible: EligibleFeatureList
    ml_config: MLConfig
    feature_catalog_fingerprint: str


@dataclass(frozen=True, slots=True)
class FusionRuntime:
    """The hybrid this deployment may apply, whether it must, and why it may not.

    A strategy is only ever *read* here, never chosen: it comes from the locked
    TEST evaluation receipt that recorded what validation selected before TEST
    was opened, and -- for a fitted hybrid -- from the serving bundle that
    reconstruction verified against that selection.  A deployment with no such
    receipt reports the hybrid unavailable rather than defaulting to one; a
    fallback would publish a hybrid nobody selected, and it would look exactly
    like one that had been.

    Three fields, because three states have to stay distinguishable:

    * :attr:`selected_strategy` -- what Phase 5 froze.  ``None`` means nothing
      qualified, which is a measured outcome.
    * :attr:`required` -- whether readiness depends on the hybrid working.  True
      whenever a hybrid *was* frozen, including when the lineage is too ambiguous
      to name which.  A runtime that cannot execute its selected hybrid is not a
      complete deployment and does not get to report itself ready.
    * :attr:`strategy` -- what will actually execute.  Either the selected
      strategy or nothing.
    """

    selected_strategy: FusionStrategy | None
    strategy: FusionStrategy | None
    stacked_state: StackedFusionState | None
    required: bool
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        """Refuse a runtime that would execute anything but the frozen hybrid."""
        if self.strategy is not None:
            if self.strategy is not self.selected_strategy:
                raise ValueError(
                    "a serving runtime executes the frozen strategy or none; "
                    "substituting one would deploy a hybrid nobody selected"
                )
            if self.unavailable_reason is not None:
                raise ValueError("an executable hybrid names no unavailable reason")
            if (self.strategy is FusionStrategy.STACKED) != (
                self.stacked_state is not None
            ):
                raise ValueError(
                    "a stacked hybrid runs its fitted meta-learner and a boolean "
                    "gate runs none; there is no default stacker"
                )
        elif self.unavailable_reason is None:
            raise ValueError("an unavailable hybrid must name a stable reason")
        if self.selected_strategy is not None and not self.required:
            raise ValueError(
                "a frozen hybrid is a required runtime component; reporting a "
                "deployment ready while its selected hybrid cannot run would "
                "call a broken system healthy"
            )

    @property
    def available(self) -> bool:
        """Return whether a hybrid verdict can be produced."""
        return self.strategy is not None


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """Everything one serving process resolved at startup, and its readiness."""

    settings: APISettings
    components: tuple[ComponentReport, ...]
    feature_config: FeatureConfig | None = None
    feature_catalog: FeatureCatalog | None = None
    rule: RuleRuntime | None = None
    ml: MLRuntime | None = None
    fusion: FusionRuntime = FusionRuntime(
        selected_strategy=None,
        strategy=None,
        stacked_state=None,
        required=False,
        unavailable_reason="no_fusion_selection",
    )
    pseudonymizer: PseudonymService | None = None

    @property
    def ready(self) -> bool:
        """Return whether every required component is ready."""
        return all(
            item.state is ComponentState.READY
            for item in self.components
            if item.required
        )

    def component(self, name: str) -> ComponentReport | None:
        """Return the report for *name*, or ``None`` when it is not tracked."""
        return next((item for item in self.components if item.component == name), None)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _report(
    name: str,
    *,
    ready: bool,
    reason: str | None = None,
    required: bool = True,
    disabled: bool = False,
) -> ComponentReport:
    """Return one component report, in the state the arguments describe."""
    if disabled:
        state = ComponentState.DISABLED
    elif ready:
        state = ComponentState.READY
    else:
        state = ComponentState.UNAVAILABLE
    return ComponentReport(
        component=name,
        state=state,
        reason=None if state is ComponentState.READY else reason,
        required=required,
    )


def build_runtime(settings: APISettings) -> RuntimeState:
    """Resolve, verify, and assemble the serving runtime.

    Never raises.  Each stage either succeeds or records a component as
    unavailable with a stable reason code, so a misconfigured deployment answers
    ``/health`` and ``/version``, reports precisely what is missing on
    ``/ready``, and refuses detection -- rather than crashing on startup with a
    traceback nobody can read from a container log.

    The order is deliberate: the feature contract is resolved before the rule
    layer and before the champion, because both are defined against it.
    """
    components: list[ComponentReport] = []

    feature_config, feature_catalog, feature_reason = _load_feature_contract(settings)
    components.append(
        _report(
            COMPONENT_FEATURE_CONTRACT,
            ready=feature_catalog is not None,
            reason=feature_reason,
        )
    )

    rule, rule_reason = _build_rule_runtime(settings, feature_catalog)
    components.append(
        _report(COMPONENT_RULE_ENGINE, ready=rule is not None, reason=rule_reason)
    )

    artifacts_ready, artifacts_reason = _inspect_model_artifacts(settings)
    components.append(
        _report(
            COMPONENT_MODEL_ARTIFACTS,
            ready=artifacts_ready,
            reason=artifacts_reason,
            required=settings.require_ml_champion,
            disabled=not settings.require_ml_champion and not artifacts_ready,
        )
    )

    ml: MLRuntime | None = None
    ml_reason: str | None = "ml_champion_disabled"
    if settings.require_ml_champion:
        ml, ml_reason = _build_ml_runtime(settings, feature_catalog)
    components.append(
        _report(
            COMPONENT_ML_CHAMPION,
            ready=ml is not None,
            reason=ml_reason,
            required=settings.require_ml_champion,
            disabled=not settings.require_ml_champion,
        )
    )

    fusion = _resolve_fusion(settings, ml)
    components.append(
        _report(
            COMPONENT_FUSION,
            ready=fusion.available,
            reason=fusion.unavailable_reason,
            # Required exactly when validation froze a hybrid. A deployment whose
            # selected hybrid cannot execute is broken, and reporting it ready
            # would hide that behind two arms that happen to work.
            required=fusion.required,
            # "Disabled" is reserved for the honest negative: no hybrid qualified,
            # so there is nothing to run. A hybrid that was selected and cannot be
            # loaded is unavailable, which is a different word on purpose.
            disabled=not fusion.available and not fusion.required,
        )
    )

    state = RuntimeState(
        settings=settings,
        components=tuple(components),
        feature_config=feature_config,
        feature_catalog=feature_catalog,
        rule=rule,
        ml=ml,
        fusion=fusion,
        pseudonymizer=_load_pseudonymizer(),
    )
    _log.info(
        "serving runtime initialised",
        ready=state.ready,
        components={item.component: str(item.state) for item in state.components},
    )
    return state


def _load_feature_contract(
    settings: APISettings,
) -> tuple[FeatureConfig | None, FeatureCatalog | None, str | None]:
    """Load the Phase 3 feature configuration and build its executable catalog."""
    try:
        config = (
            FeatureConfig()
            if settings.feature_config_path is None
            else load_feature_config(settings.feature_config_path)
        )
    except (PasswordAttackDetectorError, OSError, ValueError) as exc:
        _log.warning("feature configuration unreadable", **sanitize(exc))
        return (None, None, "feature_config_unreadable")
    try:
        return (config, build_catalog(config), None)
    except (PasswordAttackDetectorError, ValueError) as exc:
        _log.warning("feature catalog unavailable", **sanitize(exc))
        return (config, None, "feature_catalog_unavailable")


def _build_rule_runtime(
    settings: APISettings, feature_catalog: FeatureCatalog | None
) -> tuple[RuleRuntime | None, str | None]:
    """Prepare every enabled rule once, against the resolved feature catalog."""
    if feature_catalog is None:
        return (None, "feature_contract_unavailable")
    try:
        config = (
            DetectionConfig()
            if settings.detection_config_path is None
            else load_detection_config(settings.detection_config_path)
        )
    except (PasswordAttackDetectorError, OSError, ValueError) as exc:
        _log.warning("detection configuration unreadable", **sanitize(exc))
        return (None, "detection_config_unreadable")
    try:
        engine = DetectionEngine(config, feature_catalog=feature_catalog)
    except (PasswordAttackDetectorError, ValueError) as exc:
        _log.warning("rule engine preparation failed", **sanitize(exc))
        return (None, "rule_engine_preparation_failed")
    return (
        RuleRuntime(
            config=config,
            engine=engine,
            scorer=RiskScorer(config),
            catalog=RULE_CATALOG,
        ),
        None,
    )


def _inspect_model_artifacts(settings: APISettings) -> tuple[bool, str | None]:
    """Report whether the configured artifact root holds a frozen champion.

    Structural only: does the root exist, and is a champion lock present under
    it.  Whether that lock *verifies* is the ``ml_champion`` component's job, and
    keeping the two separate lets a readiness document distinguish "nothing was
    ever frozen here" from "something was frozen and no longer checks out".
    """
    root = settings.artifact_root
    if root is None:
        return (False, "artifact_root_not_configured")
    if not root.is_dir():
        return (False, "artifact_root_not_found")
    champion_root = root / "champion"
    if not champion_root.is_dir():
        return (False, "no_champion_frozen")
    locks = [
        directory
        for directory in champion_root.iterdir()
        if directory.is_dir() and (directory / "champion.lock").is_file()
    ]
    if not locks:
        return (False, "no_champion_frozen")
    return (True, None)


def _build_ml_runtime(
    settings: APISettings, feature_catalog: FeatureCatalog | None
) -> tuple[MLRuntime | None, str | None]:
    """Verify the frozen champion end to end and bind it to the feature contract.

    Every step is the existing verification: the champion loader checks the
    freeze receipt, the selection, the run, the artifact bytes, the preprocessor,
    the calibrator, the operating point and the dependency ranges; the feature
    contract check then confirms that the rows this build will compute come from
    the contract the champion was frozen against.  Nothing is skipped when an
    input is absent -- an absent input is a refusal.
    """
    if feature_catalog is None:
        return (None, "feature_contract_unavailable")
    if settings.artifact_root is None:
        return (None, "artifact_root_not_configured")
    if settings.allowlist_path is None:
        return (None, "allowlist_not_configured")

    try:
        ml_config = (
            MLConfig()
            if settings.ml_config_path is None
            else load_ml_config(settings.ml_config_path)
        )
    except (PasswordAttackDetectorError, OSError, ValueError) as exc:
        _log.warning("ML configuration unreadable", **sanitize(exc))
        return (None, "ml_config_unreadable")

    try:
        allowlist = load_feature_allowlist(settings.allowlist_path)
        eligible = resolve_eligible_features(
            feature_catalog,
            allowlist,
            include_leakage_classes=ml_config.preprocessing.include_leakage_classes,
            include_feature_groups=ml_config.preprocessing.include_feature_groups,
            feature_schema_version=ml_config.required_feature_schema_version,
        )
    except (PasswordAttackDetectorError, OSError, ValueError) as exc:
        _log.warning("feature allowlist unusable", **sanitize(exc))
        return (None, "allowlist_unusable")

    root = settings.artifact_root
    try:
        champion = FrozenChampion.load(
            root,
            ledger=ExperimentLedger(root / "ledger"),
            scope_key=settings.champion_scope_key,
            config=ml_config,
        )
    except (PasswordAttackDetectorError, OSError, ValueError) as exc:
        _log.warning("frozen champion unavailable", **sanitize(exc))
        return (None, "champion_verification_failed")

    try:
        verify_inference_feature_contract(
            champion.lock,
            # The rows this service scores are computed here, by this build's own
            # feature engine, so the "manifest" describing them is this build's
            # own declaration rather than a file read off disk. The two checks it
            # feeds are therefore trivially satisfied; the four that follow --
            # against the reviewed allowlist and against the champion lock -- are
            # the ones that bite, and they are the ones that matter.
            feature_manifest={
                "feature_catalog_fingerprint": feature_catalog.fingerprint(),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
            },
            catalog_fingerprint=feature_catalog.fingerprint(),
            allowlist_fingerprint=allowlist.fingerprint(),
            eligible_feature_list_fingerprint=eligible.fingerprint(),
            required_feature_schema_version=ml_config.required_feature_schema_version,
            compatible_catalog_fingerprints=(
                allowlist.compatible_feature_catalog_fingerprints
            ),
        )
    except PasswordAttackDetectorError as exc:
        _log.warning("serving feature contract mismatch", **sanitize(exc))
        return (None, "feature_contract_mismatch")

    return (
        MLRuntime(
            champion=champion,
            eligible=eligible,
            ml_config=ml_config,
            feature_catalog_fingerprint=feature_catalog.fingerprint(),
        ),
        None,
    )


def _unavailable_fusion(
    reason: str, *, required: bool, selected: FusionStrategy | None = None
) -> FusionRuntime:
    """Return a runtime that executes no hybrid, for the stated reason.

    *selected* is carried through when it is known, so a deployment that cannot
    load the hybrid it selected still reports **which** hybrid that was.  Without
    it, "the stacker is missing" and "nothing qualified" would look identical from
    outside, and only one of them is somebody's problem to fix.
    """
    return FusionRuntime(
        selected_strategy=selected,
        strategy=None,
        stacked_state=None,
        required=required,
        unavailable_reason=reason,
    )


def _resolve_fusion(settings: APISettings, ml: MLRuntime | None) -> FusionRuntime:
    """Read the frozen hybrid, load what it needs to run, or say why it cannot.

    The strategy lives in the locked TEST evaluation receipt, which records what
    was selected on validation-B *before* TEST was opened.  What each strategy
    then needs to be executable differs, and so does what its absence means:

    * ``or_gate`` and ``and_gate`` are functions of the two booleans and need no
      fitted artifact.  A published serving bundle still *binds* the selection --
      when one is present it is verified and must agree with the receipt -- but a
      gate deployment without one runs the gate the receipt names.
    * ``stacked`` is a fitted meta-learner.  It runs only from a materialized
      serving bundle whose state reconstruction recomputed the fingerprint Phase 5
      sealed.  No bundle, an unverifiable bundle, or a bundle for a different
      lineage each leave the hybrid unavailable -- and, because a hybrid *was*
      selected, leave the deployment **not ready**.

    Nothing here fits anything.  The bundle is loaded and verified; there is no
    code path from this function to the stacker's fit, and an import-time guard at
    the bottom of this module refuses one.

    Fail-closed throughout: a receipt this process cannot read might be the one
    naming a hybrid, so it is treated as a required component that failed rather
    than as an absent one.
    """
    if ml is None or settings.artifact_root is None:
        return _unavailable_fusion("ml_champion_unavailable", required=False)

    receipts_root = settings.artifact_root / EVALUATIONS_DIR
    if not receipts_root.is_dir():
        return _unavailable_fusion("no_fusion_selection", required=False)

    lock_fingerprint = ml.champion.lock.lock_fingerprint
    selected: set[FusionStrategy] = set()
    for directory in sorted(receipts_root.iterdir()):
        receipt = directory / EVALUATION_RECEIPT_FILE
        if not directory.is_dir() or not receipt.is_file():
            continue
        try:
            record = TestEvaluationRecord.from_json(receipt.read_text(encoding="utf-8"))
        except (PasswordAttackDetectorError, OSError, ValueError) as exc:
            # A receipt that does not verify may be the one that selected a
            # hybrid. Treating it as "nothing was selected" would let a corrupted
            # deployment present as a complete rule-and-model one.
            _log.warning("evaluation receipt unreadable", **sanitize(exc))
            return _unavailable_fusion("evaluation_receipt_unreadable", required=True)
        if record.champion_lock_fingerprint != lock_fingerprint:
            continue
        if record.selected_fusion_strategy is not None:
            selected.add(record.selected_fusion_strategy)

    if not selected:
        # The honest negative: nothing qualified on validation-B, so there is no
        # hybrid to run and none to require. See ``system_status_document``.
        return _unavailable_fusion("no_fusion_selection", required=False)
    if len(selected) > 1:
        # Two receipts for one champion naming different strategies is not a
        # tie to break; it is a lineage nobody can attribute a verdict to.
        return _unavailable_fusion("ambiguous_fusion_selection", required=True)

    strategy = selected.pop()
    bundle, bundle_reason = _load_bundle(settings, ml)
    if bundle_reason is not None:
        if strategy is FusionStrategy.STACKED:
            return _unavailable_fusion(bundle_reason, required=True, selected=strategy)
        # A gate needs no fitted artifact. An *unverifiable* bundle is still a
        # refusal, because a bundle that does not check out may be describing a
        # different lineage; only a genuinely absent one is an ordinary state.
        if bundle_reason != "serving_bundle_not_published":
            return _unavailable_fusion(bundle_reason, required=True, selected=strategy)
        return FusionRuntime(
            selected_strategy=strategy,
            strategy=strategy,
            stacked_state=None,
            required=True,
            unavailable_reason=None,
        )

    assert bundle is not None  # a bundle or a reason, never neither
    if bundle.strategy is not strategy:
        # The receipt and the bundle disagree about what was selected. One of
        # them is describing a deployment this is not.
        return _unavailable_fusion(
            "fusion_selection_conflict", required=True, selected=strategy
        )
    return FusionRuntime(
        selected_strategy=strategy,
        strategy=strategy,
        stacked_state=bundle.stacked_state,
        required=True,
        unavailable_reason=None,
    )


def _load_bundle(
    settings: APISettings, ml: MLRuntime
) -> tuple[ServingBundle | None, str | None]:
    """Load and verify the serving bundle for this champion scope.

    Loading is verification: the manifest's own seal, every payload digest, the
    frozen selection's seal, and -- for a stacked bundle -- that the state is the
    one the frozen selection named.  Nothing partially verified is returned.
    """
    root = settings.bundle_root
    if root is None:
        return (None, "serving_bundle_not_published")
    try:
        bundle = load_serving_bundle(root, scope_key=ml.champion.lock.scope_key)
    except ArtifactNotFoundError:
        return (None, "serving_bundle_not_published")
    except (PasswordAttackDetectorError, OSError, ValueError) as exc:
        _log.warning("serving bundle unverifiable", **sanitize(exc))
        return (None, "serving_bundle_unverifiable")
    if bundle.manifest.champion_lock_fingerprint != ml.champion.lock.lock_fingerprint:
        return (None, "serving_bundle_lineage_mismatch")
    return (bundle, None)


def _load_pseudonymizer() -> PseudonymService | None:
    """Return the keyed pseudonym service, or ``None`` when no key is configured.

    The key is read only from ``PAD_PSEUDONYMIZATION_KEY`` or the untracked
    ``.env`` file, exactly as the ingestion adapters read it.  It is never
    logged, never serialised, and never reaches a response.
    """
    try:
        return PseudonymService.from_settings(load_settings())
    except (PseudonymizationError, PasswordAttackDetectorError, ValueError):
        # No key is an ordinary deployment state: a caller sending pseudonymous
        # identifiers needs none. Only a caller sending a raw address does, and
        # that request is refused with its own stable code.
        return None


# ---------------------------------------------------------------------------
# Operational documents
# ---------------------------------------------------------------------------


def readiness_document(runtime: RuntimeState) -> ReadinessResponse:
    """Return the aggregate readiness of every runtime component."""
    return ReadinessResponse(
        status=ReadinessState.READY if runtime.ready else ReadinessState.NOT_READY,
        version=__version__,
        components=runtime.components,
    )


def version_document() -> VersionResponse:
    """Return every contract version this build implements.

    Constants read off the modules that own them, so a version cannot drift from
    the contract it names.  Nothing host-specific is present.
    """
    return VersionResponse(
        package_version=__version__,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        detection_schema_version=DETECTION_SCHEMA_VERSION,
        scoring_version=SCORING_VERSION,
        ml_schema_version=ML_SCHEMA_VERSION,
        fusion_schema_version=FUSION_SCHEMA_VERSION,
    )


def system_status_document(runtime: RuntimeState) -> SystemStatusResponse:
    """Return which detection layers this deployment is running."""
    rule = runtime.rule
    return SystemStatusResponse(
        status=ReadinessState.READY if runtime.ready else ReadinessState.NOT_READY,
        package_version=__version__,
        rule_detection_enabled=rule is not None,
        ml_detection_enabled=runtime.ml is not None,
        hybrid_detection_enabled=runtime.fusion.available,
        fusion_strategy=runtime.fusion.strategy,
        # What validation selected, reported separately from what is running: a
        # deployment whose frozen hybrid cannot execute must say both, or the
        # absence of a hybrid reads as "none qualified" when it means "the one
        # that qualified is missing".
        frozen_fusion_strategy=runtime.fusion.selected_strategy,
        hybrid_required=runtime.fusion.required,
        stacked_state_fingerprint=(
            None
            if runtime.fusion.stacked_state is None
            else runtime.fusion.stacked_state.state_fingerprint
        ),
        fusion_unavailable_reason=runtime.fusion.unavailable_reason,
        champion_model_family=(
            None if runtime.ml is None else str(runtime.ml.champion.lock.model_family)
        ),
        enabled_rule_count=0 if rule is None else len(rule.engine.enabled_rule_ids),
        registered_rule_count=len(RULE_CATALOG),
        max_batch_events=runtime.settings.max_batch_events,
    )


def model_info_document(runtime: RuntimeState) -> ModelInfoResponse:
    """Return the champion's public-safe identity, lineage, and operating point.

    Identity and decision semantics only.  No coefficient, tree array, feature
    value, artifact path, or entity pseudonym is reachable from this document.
    """
    ml = runtime.ml
    if ml is None:
        report = runtime.component(COMPONENT_ML_CHAMPION)
        return ModelInfoResponse(
            available=False,
            unavailable_reason=(
                "ml_champion_unavailable"
                if report is None or report.reason is None
                else report.reason
            ),
        )
    lock = ml.champion.lock
    return ModelInfoResponse(
        available=True,
        model_family=str(lock.model_family),
        catalog_model_id=lock.catalog_model_id,
        model_id=lock.model_id,
        task=str(lock.task),
        score_kind=ml.champion.score_kind,
        calibrated=is_probability(ml.champion.score_kind),
        decision_threshold=ml.champion.decision_threshold,
        champion_scope_key=ml.champion.scope_key,
        freeze_record_id=ml.champion.freeze_record_id,
        training_run_id=lock.training_run_id,
        validation_selection_id=lock.validation_selection_id,
        ml_schema_version=ML_SCHEMA_VERSION,
        required_feature_schema_version=ml.ml_config.required_feature_schema_version,
        category_head_available=ml.champion.category is not None,
    )


def rule_catalog_document(runtime: RuntimeState) -> RuleCatalogResponse:
    """Return the public rule catalog and which rules this deployment enabled."""
    rule = runtime.rule
    catalog = RULE_CATALOG if rule is None else rule.catalog
    rules = tuple(
        RuleSummary(
            rule_id=spec.rule_id,
            rule_version=spec.rule_version,
            name=spec.display_name,
            description=spec.description,
            family=spec.family,
            attack_category=spec.attack_category,
            default_severity=spec.default_severity,
            enabled=rule is not None and rule.config.is_enabled(spec.rule_id),
            deprecated=spec.deprecated,
        )
        for spec in catalog.specs
    )
    return RuleCatalogResponse(
        detection_schema_version=DETECTION_SCHEMA_VERSION,
        rule_count=len(rules),
        enabled_rule_count=sum(1 for item in rules if item.enabled),
        rules=rules,
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_single(
    runtime: RuntimeState, request: WindowRequestBase
) -> DetectionResponse:
    """Score a window and return the verdict for the one selected anchor.

    Raises:
        APIError: with :attr:`ErrorCode.ANCHOR_SELECTION_ERROR` when the request
            selects anything other than exactly one anchor, and with whichever
            code the underlying stage refused under.
    """
    anchors = request.resolved_anchor_ids()
    if len(anchors) != 1:
        raise APIError(
            ErrorCode.ANCHOR_SELECTION_ERROR,
            detail={"selected_anchor_count": len(anchors), "expected": 1},
        )
    window, detections = _evaluate(runtime, request, anchors)
    return DetectionResponse(window=window, anchor=detections[0])


def detect_batch(
    runtime: RuntimeState, request: WindowRequestBase
) -> BatchDetectionResponse:
    """Score a window and return a verdict per selected anchor, in canonical order."""
    anchors = request.resolved_anchor_ids()
    window, detections = _evaluate(runtime, request, anchors)
    return BatchDetectionResponse(window=window, anchors=detections)


def _evaluate(
    runtime: RuntimeState,
    request: WindowRequestBase,
    anchor_ids: Sequence[str],
) -> tuple[WindowSummary, tuple[AnchorDetection, ...]]:
    """Run the whole composition once and project it onto the selected anchors."""
    if not runtime.ready or runtime.rule is None or runtime.feature_config is None:
        raise APIError(ErrorCode.RUNTIME_NOT_READY)
    if len(request.events) > runtime.settings.max_batch_events:
        raise APIError(
            ErrorCode.BATCH_LIMIT_EXCEEDED,
            detail={
                "event_count": len(request.events),
                "max_batch_events": runtime.settings.max_batch_events,
            },
        )

    events = _canonical_events(runtime, request)
    rows = _feature_rows(runtime, events)
    assessments = _rule_assessments(runtime, rows)
    predictions = _ml_predictions(runtime, rows)

    ordered = sorted(anchor_ids, key=lambda anchor: _sort_key(assessments, anchor))
    detections = tuple(
        _anchor_detection(runtime, assessments[anchor], predictions.get(anchor))
        for anchor in ordered
    )
    summary = WindowSummary(
        event_count=len(events),
        anchor_count=len(detections),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        detection_schema_version=DETECTION_SCHEMA_VERSION,
        enabled_rule_count=len(runtime.rule.engine.enabled_rule_ids),
        evaluated_snapshot_count=len(rows),
    )
    return (summary, detections)


def _sort_key(
    assessments: Mapping[str, RiskAssessment], anchor: str
) -> tuple[datetime, str]:
    """Return the canonical ordering key for one anchor."""
    return (assessments[anchor].anchor_event_time, anchor)


def _canonical_events(
    runtime: RuntimeState, request: WindowRequestBase
) -> tuple[AuthEvent, ...]:
    """Convert the request into canonical events, pseudonymizing any raw address.

    A raw ``source_ip`` is turned into a source pseudonym by the project's keyed
    HMAC service and then dropped: the address itself is never stored on the
    event, never logged, and never returned.
    """
    converted: list[AuthEvent] = []
    for event in request.events:
        if event.source_id is not None:
            source_id = event.source_id
        else:
            if runtime.pseudonymizer is None:
                raise APIError(ErrorCode.PSEUDONYMIZATION_UNAVAILABLE)
            assert event.source_ip is not None  # guaranteed by the request schema
            source_id = runtime.pseudonymizer.pseudonymize("source", event.source_ip)
        try:
            converted.append(event.to_canonical_event(source_id=source_id))
        except ValueError as exc:
            # Pydantic's ValidationError is a ValueError; its message can quote
            # the offending value, so it is logged by type alone and never
            # forwarded.
            _log.info("rejected an invalid authentication event", **sanitize(exc))
            raise APIError(ErrorCode.INVALID_AUTHENTICATION_EVENT) from None
    return tuple(converted)


def _feature_rows(
    runtime: RuntimeState, events: Sequence[AuthEvent]
) -> tuple[dict[str, Any], ...]:
    """Compute one point-in-time feature snapshot per event.

    A fresh :class:`FeatureEngine` per request: the engine carries per-run entity
    state, and a shared instance would make two concurrent windows visible to one
    another's history -- which is exactly the leakage the point-in-time contract
    exists to prevent.
    """
    assert runtime.feature_config is not None  # checked by the caller
    try:
        engine = FeatureEngine(runtime.feature_config, runtime.feature_catalog)
        return engine.run(list(events)).rows
    except PasswordAttackDetectorError as exc:
        _log.warning("feature computation failed", **sanitize(exc))
        raise APIError(ErrorCode.FEATURE_CONTRACT_FAILURE) from None


def _rule_assessments(
    runtime: RuntimeState, rows: Sequence[Mapping[str, Any]]
) -> dict[str, RiskAssessment]:
    """Run every enabled rule over the window and score each anchor's risk."""
    assert runtime.rule is not None  # checked by the caller
    try:
        results = runtime.rule.engine.run_diagnostic(list(rows))
        scored = runtime.rule.scorer.score(
            results,
            evaluated_anchors={
                str(row[ANCHOR_EVENT_ID]): row[ANCHOR_EVENT_TIME] for row in rows
            },
        )
    except PasswordAttackDetectorError as exc:
        _log.warning("rule evaluation failed", **sanitize(exc))
        raise APIError(ErrorCode.RULE_DETECTION_UNAVAILABLE) from None
    return {item.anchor_event_id: item for item in scored.assessments}


def _ml_predictions(
    runtime: RuntimeState, rows: Sequence[Mapping[str, Any]]
) -> dict[str, BinaryPrediction]:
    """Score the window under the frozen champion, or return nothing at all.

    ``predict_serving_binary`` owns every step of this: the reviewed feature
    order, the frozen preprocessor, the project-owned inference adapter, the
    frozen calibrator where the lineage has one, and the frozen operating point
    applied to whichever score it was selected against.  It shares that whole
    decision with the published-split predictor -- one implementation, called
    from two places -- so a live row and an evaluated row cannot be scored by two
    subtly different rules.

    ``assemble_serving_batch`` takes no split table and no scope argument, so
    there is nowhere in this function for a live request to acquire an
    experimental population.
    """
    ml = runtime.ml
    if ml is None:
        return {}
    try:
        batch = assemble_serving_batch(
            feature_rows=list(rows),
            eligible=ml.eligible,
            feature_catalog_fingerprint=ml.feature_catalog_fingerprint,
        )
        predictions = predict_serving_binary(ml.champion, batch)
    except PasswordAttackDetectorError as exc:
        _log.warning("model scoring failed", **sanitize(exc))
        raise APIError(ErrorCode.ML_CHAMPION_UNAVAILABLE) from None
    return {item.anchor_event_id: item for item in predictions}


def _anchor_detection(
    runtime: RuntimeState,
    assessment: RiskAssessment,
    prediction: BinaryPrediction | None,
) -> AnchorDetection:
    """Assemble one anchor's three-layer verdict, keeping the layers separate."""
    rule_layer = RuleLayerResult(
        flagged=assessment.fired_rule_count > 0,
        risk_score=assessment.risk_score,
        severity=assessment.severity,
        primary_attack_category=assessment.primary_attack_category,
        contributing_categories=assessment.contributing_categories,
        fired_rule_ids=assessment.fired_rule_ids,
        fired_rule_count=assessment.fired_rule_count,
        insufficient_data_count=assessment.insufficient_data_count,
        scoring_version=assessment.scoring_version,
        evidence=assessment.top_evidence,
    )

    if prediction is None:
        report = runtime.component(COMPONENT_ML_CHAMPION)
        reason = (
            "ml_champion_unavailable"
            if report is None or report.reason is None
            else report.reason
        )
        return AnchorDetection(
            anchor_event_id=assessment.anchor_event_id,
            anchor_event_time=assessment.anchor_event_time,
            rule=rule_layer,
            ml=MLLayerResult(available=False, unavailable_reason=reason),
            hybrid=HybridLayerResult(
                available=False, unavailable_reason="ml_champion_unavailable"
            ),
            severity=assessment.severity,
        )

    return AnchorDetection(
        anchor_event_id=assessment.anchor_event_id,
        anchor_event_time=assessment.anchor_event_time,
        rule=rule_layer,
        ml=MLLayerResult(
            available=True,
            flagged=prediction.flagged_malicious,
            score_kind=prediction.score_kind,
            decision_score=prediction.malicious_decision_score,
            probability=prediction.malicious_probability,
            decision_threshold=prediction.decision_threshold,
        ),
        hybrid=hybrid_layer(runtime, assessment, prediction),
        severity=assessment.severity,
    )


def hybrid_layer(
    runtime: RuntimeState,
    assessment: RiskAssessment,
    prediction: BinaryPrediction,
) -> HybridLayerResult:
    """Apply the frozen fusion strategy, or report that none is applicable.

    :func:`~password_attack_detector.ml.fusion.fuse` performs the combination.
    It takes the rule side as a **boolean decision** and the model side as its
    own typed evidence; the rule layer's 0-100 ordinal magnitude is carried for
    reporting and never enters the arithmetic, because there is none.

    For ``stacked``, the fitted meta-learner passed here is the one the serving
    bundle published and startup verified against the frozen selection.  ``fuse``
    refuses a stacked strategy with no state, so a runtime that somehow reached
    this point without one produces a refusal rather than a fabricated verdict --
    and :class:`FusionRuntime` makes that state unconstructible in the first
    place.
    """
    if not runtime.fusion.available or runtime.fusion.strategy is None:
        return HybridLayerResult(
            available=False,
            unavailable_reason=runtime.fusion.unavailable_reason
            or "no_fusion_selection",
        )
    try:
        decision = fuse(
            runtime.fusion.strategy,
            anchor_event_id=assessment.anchor_event_id,
            anchor_event_time=assessment.anchor_event_time,
            rule=RuleEvidence(
                flagged=assessment.fired_rule_count > 0,
                ordinal_risk_score=assessment.risk_score,
            ),
            ml=MLEvidence(
                flagged=prediction.flagged_malicious,
                decision_score=prediction.malicious_decision_score,
                calibrated_probability=prediction.malicious_probability,
                score_kind=prediction.score_kind,
            ),
            stacked=runtime.fusion.stacked_state,
        )
    except (PasswordAttackDetectorError, ValueError) as exc:
        _log.warning("fusion refused a row", **sanitize(exc))
        return HybridLayerResult(
            available=False, unavailable_reason="fusion_refused_evidence"
        )
    return HybridLayerResult(
        available=True, flagged=decision.fused_flagged, strategy=decision.strategy
    )


def _assert_no_scientific_state_is_written() -> None:
    """Fail at import if this module acquires a writer for frozen scientific state.

    The serving layer reads frozen artifacts and writes none.  A publisher, a
    freezer, a selector, or a trainer imported here would be a serving process
    able to change what it serves -- so the names are refused rather than
    reviewed for.
    """
    import sys

    forbidden = {
        "build_champion_lock",
        "freeze_champion",
        "fit_stacked_fusion",
        "materialize_serving_bundle",
        "prepare_fusion_selection",
        "publish_predictions",
        "reconstruct_stacked_state",
        "select_binary_threshold",
        "select_fusion_strategy",
        "train_model",
        "write_serving_bundle",
    }
    offending = sorted(forbidden & set(vars(sys.modules[__name__])))
    if offending:
        raise ValueError(
            f"{__name__} imported {offending}, which would let a serving process "
            f"change the frozen state it exists to serve"
        )


_assert_no_scientific_state_is_written()


def _assert_serving_names_no_split() -> None:
    """Fail at import if a split label becomes reachable from the serving path.

    The property is that a live request is never filed as an experimental
    population.  What would quietly undo it is one import: ``MLSplit`` back in
    this namespace, and the next scoring change reaches for ``MLSplit.TEST``
    because the assembler used to want one.  So the name is refused, and the
    serving scope is asserted to be the type that is not a split.
    """
    import sys

    namespace = vars(sys.modules[__name__])
    forbidden = {"MLSplit", "SplitRow", "assemble_inference_dataset", "predict_binary"}
    offending = sorted(forbidden & set(namespace))
    if offending:
        raise ValueError(
            f"{__name__} imported {offending}; live serving rows belong to no "
            f"split and must not be scored through the published-split path"
        )
    if not isinstance(SERVING_SCOPE, ServingScope):
        raise ValueError(
            "the serving scope must be a ServingScope; a split value here would "
            "file live traffic as an evaluation population"
        )


_assert_serving_names_no_split()
