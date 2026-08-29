"""Typer CLI for deployment artifacts.

Two commands, and neither of them decides anything::

    password-attack-detector deploy materialize   reconstruct, verify, publish
    password-attack-detector deploy inspect       load and verify what is published

``materialize`` is the **offline** step that makes a frozen stacked hybrid
servable.  It is deliberately a separate operator action rather than something
the API does at startup: fitting a meta-learner is not a thing a serving process
should be able to do, and the only way to make that true is for the serving
process to have no path to it.

**No command here prints an identifier, a path, or a measured figure.**  What it
prints is which strategy is frozen, whether the reconstruction agreed with the
sealed fingerprint, and where the bundle went -- rendered relatively, like every
other command in this project.

Heavy imports live inside the command bodies so ``--help`` stays fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from password_attack_detector.exceptions import (
    ArtifactNotFoundError,
    ConfigurationError,
    DataValidationError,
    ExperimentPublicationError,
    LedgerConflictError,
    ManifestVerificationError,
    MLConfigurationError,
    ModelNotReadyError,
    ModelSerializationError,
    ModelTrainingError,
)

deployment_app = typer.Typer(
    name="deploy",
    help=(
        "Deployment artifacts: materialize the frozen hybrid strategy into a "
        "serving bundle and inspect what is published. Nothing here selects, "
        "retrains, or re-thresholds anything -- a reconstruction that does not "
        "recompute the fingerprint Phase 5 sealed is refused."
    ),
    no_args_is_help=True,
)

_console = Console()
_err = Console(stderr=True)

#: Exceptions whose message is safe to show the user, exactly as ``ml`` treats
#: them: every one is raised by this project with a message written to carry
#: counts and declared names only.
_REPORTABLE = (
    ArtifactNotFoundError,
    ConfigurationError,
    DataValidationError,
    ExperimentPublicationError,
    LedgerConflictError,
    MLConfigurationError,
    ManifestVerificationError,
    ModelNotReadyError,
    ModelSerializationError,
    ModelTrainingError,
)

#: What each refusal means, in one line, for an operator reading a terminal.
_REFUSALS: dict[str, str] = {
    "rule_configuration_mismatch": (
        "the rule configuration given here is not the one the frozen selection "
        "was made against"
    ),
    "stacked_state_not_reconstructible": (
        "the out-of-fold meta-features could not be rebuilt from these inputs, so "
        "there is no stacker to publish"
    ),
    "fusion_selection_mismatch": (
        "the reconstruction did not reproduce the frozen selection; at least one "
        "semantic upstream input differs from the one Phase 5 saw"
    ),
    "stacked_state_fingerprint_mismatch": (
        "the reconstructed stacker is not the stacker the frozen selection named"
    ),
}


def _display(path: Path) -> str:
    """Render a path relative to the working directory where possible."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _fail(message: str) -> None:
    """Print an error and exit non-zero, without chaining a traceback."""
    _err.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=1) from None


def _guard(action: str, call: Any) -> Any:
    """Run *call*, converting a known failure into a sanitized exit."""
    try:
        return call()
    except _REPORTABLE as exc:
        _fail(f"{action}: {exc}")
    except Exception as exc:
        _fail(f"{action} failed ({type(exc).__name__})")


def _artifacts_root() -> Path:
    """Return the default artifact root, matching the ML layer's default."""
    from password_attack_detector.paths import get_artifacts_dir

    return get_artifacts_dir()


@deployment_app.command()
def materialize(
    features_path: Annotated[
        Path, typer.Option("--features", help="Phase 3 feature snapshots Parquet file.")
    ],
    labels_path: Annotated[
        Path, typer.Option("--labels", help="Phase 3 feature labels Parquet file.")
    ],
    splits_path: Annotated[
        Path, typer.Option("--splits", help="Phase 3 feature splits Parquet file.")
    ],
    allowlist_path: Annotated[
        Path,
        typer.Option("--allowlist", help="Reviewed ML feature allowlist YAML file."),
    ],
    risk_path: Annotated[
        Path,
        typer.Option(
            "--risk-assessments",
            help="Phase 4 risk assessments from the published detection run the "
            "frozen selection was made against.",
        ),
    ],
    validation_prediction: Annotated[
        str,
        typer.Option(
            "--validation-prediction",
            help="The published validation prediction the hybrid was selected on.",
        ),
    ],
    campaign_labels: Annotated[
        Path | None,
        typer.Option("--campaign-labels", help="Phase 2 label table."),
    ] = None,
    validation_risk_path: Annotated[
        Path | None,
        typer.Option(
            "--validation-risk-assessments",
            help="Additional Phase 4 risk assessments for the validation anchors.",
        ),
    ] = None,
    rule_config_path: Annotated[
        Path | None,
        typer.Option("--rule-config", help="Frozen Phase 4 detection YAML."),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="ML YAML configuration file.")
    ] = None,
    feature_config_path: Annotated[
        Path | None,
        typer.Option("--feature-config", help="Phase 3 feature YAML configuration."),
    ] = None,
    output_root: Annotated[
        Path | None,
        typer.Option(
            "--output-root",
            "-o",
            help="Root holding the champion, the predictions, the ledger, and "
            "the evaluations. The bundle is written under it.",
        ),
    ] = None,
    scope_key: Annotated[
        str | None,
        typer.Option("--scope-key", help="Which frozen champion scope to deploy."),
    ] = None,
) -> None:
    """Make the frozen hybrid strategy servable, or refuse and say why.

    **Every input is pre-TEST lineage.**  Feature snapshots, split assignments,
    the reviewed allowlist, the published validation prediction, the frozen rule
    run, and the frozen champion: exactly what the original selection consumed.
    There is no ``--prediction`` for a TEST publication and no way to open a TEST
    label, because the reconstruction is a reproduction of a decision that was
    made before TEST existed.

    For ``or_gate`` or ``and_gate`` this publishes the frozen selection, which is
    all a boolean gate needs to be executed from frozen state rather than from
    configuration.  For ``stacked`` it additionally refits the meta-learner from
    out-of-fold TRAIN evidence and **refuses to publish** unless the result
    recomputes both the frozen selection's fingerprint and the frozen stacked
    state's.

    Nothing is overwritten. Materializing an identical bundle twice writes
    nothing the second time.
    """
    from password_attack_detector.deployment.materialize import (
        materialize_serving_bundle,
    )
    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.config import (
        FeatureConfig,
        load_feature_config,
    )
    from password_attack_detector.ml.config import MLConfig, load_ml_config
    from password_attack_detector.ml.dataset import load_ml_dataset
    from password_attack_detector.ml.features import (
        load_feature_allowlist,
        resolve_eligible_features,
    )
    from password_attack_detector.ml.ledger import ExperimentLedger
    from password_attack_detector.ml.predictions import FrozenChampion

    for path in (features_path, labels_path, splits_path, allowlist_path, risk_path):
        if not path.exists():
            _fail(f"Input not found: {_display(path)}")

    config: Any = _guard(
        "Cannot load the ML configuration",
        lambda: MLConfig() if config_path is None else load_ml_config(config_path),
    )
    feature_config = _guard(
        "Cannot load the feature configuration",
        lambda: (
            FeatureConfig()
            if feature_config_path is None
            else load_feature_config(feature_config_path)
        ),
    )
    catalog = _guard(
        "Cannot build the feature catalog", lambda: build_catalog(feature_config)
    )
    allowlist = _guard(
        "Cannot load the feature allowlist",
        lambda: load_feature_allowlist(allowlist_path),
    )
    eligible = _guard(
        "Cannot resolve the eligible feature set",
        lambda: resolve_eligible_features(
            catalog,
            allowlist,
            include_leakage_classes=config.preprocessing.include_leakage_classes,
            include_feature_groups=config.preprocessing.include_feature_groups,
            feature_schema_version=config.required_feature_schema_version,
        ),
    )
    dataset = _guard(
        "Cannot assemble the labelled dataset",
        lambda: load_ml_dataset(
            features_path=features_path,
            labels_path=labels_path,
            splits_path=splits_path,
            eligible=eligible,
            campaign_labels_path=campaign_labels,
            feature_catalog_fingerprint=catalog.fingerprint(),
        ),
    )

    root = output_root or (_artifacts_root() / "ml")
    champion = _guard(
        "Cannot load the frozen champion",
        lambda: FrozenChampion.load(
            root,
            ledger=ExperimentLedger(root / "ledger"),
            scope_key=scope_key,
            config=config,
        ),
    )

    rule = _rule_evidence(risk_path, "risk assessments")
    if validation_risk_path is not None:
        rule = {**rule, **_rule_evidence(validation_risk_path, "validation risk")}
    fingerprint = _rule_configuration_fingerprint(rule_config_path)
    predictions = _validation_predictions(root, validation_prediction)

    outcome = _guard(
        "Cannot materialize the serving bundle",
        lambda: materialize_serving_bundle(
            root=root,
            dataset=dataset,
            config=config,
            eligible=eligible,
            feature_catalog=catalog,
            champion=champion,
            rule=rule,
            rule_configuration_fingerprint=fingerprint,
            validation_predictions=predictions,
        ),
    )

    _console.print(_outcome_table(outcome))
    if not outcome.published:
        detail = _REFUSALS.get(outcome.refusal or "", "the frozen lineage disagreed")
        _err.print(
            f"[red]Refusing to publish a serving bundle:[/red] {detail}. Nothing "
            f"was written, and no strategy was substituted for the selected one."
        )
        raise typer.Exit(code=2)

    assert outcome.directory is not None  # published implies a directory
    verb = "Published" if outcome.created else "Already published"
    _console.print(f"{verb} under {_display(outcome.directory)}")
    _console.print(
        "[dim]The serving layer loads and verifies this bundle. It never fits "
        "the stacker, and it never substitutes a strategy for the frozen "
        "one.[/dim]"
    )


def _outcome_table(outcome: Any) -> Table:
    """Return what the materialization established, without an identifier."""
    table = Table(title="Serving bundle", show_header=False, box=None)
    table.add_row("Frozen strategy", str(outcome.strategy))
    if outcome.frozen_state_fingerprint is None:
        table.add_row("Fitted state", "not required by this strategy")
    else:
        table.add_row("Frozen state digest", outcome.frozen_state_fingerprint[:16])
        table.add_row(
            "Reconstructed digest",
            (
                "not reached"
                if outcome.reconstructed_state_fingerprint is None
                else outcome.reconstructed_state_fingerprint[:16]
            ),
        )
        table.add_row(
            "Fingerprints agree", "yes" if outcome.fingerprints_agree else "NO"
        )
    table.add_row("Published", "yes" if outcome.published else "no")
    return table


@deployment_app.command()
def inspect(
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", "-o", help="Root holding the serving bundle."),
    ] = None,
    scope_key: Annotated[
        str | None,
        typer.Option("--scope-key", help="Which frozen champion scope to inspect."),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="ML YAML configuration file.")
    ] = None,
) -> None:
    """Load and fully verify the published serving bundle.

    Exactly the verification the serving runtime performs at startup, run from a
    terminal: the manifest's seal, every payload digest, the frozen selection's
    seal, and -- for ``stacked`` -- that the bundled state is the state the frozen
    selection named. A bundle that fails any of it exits non-zero.
    """
    from password_attack_detector.deployment.bundle import load_serving_bundle
    from password_attack_detector.ml.config import MLConfig, load_ml_config
    from password_attack_detector.ml.ledger import ExperimentLedger
    from password_attack_detector.ml.predictions import FrozenChampion

    config: Any = _guard(
        "Cannot load the ML configuration",
        lambda: MLConfig() if config_path is None else load_ml_config(config_path),
    )
    root = output_root or (_artifacts_root() / "ml")
    champion = _guard(
        "Cannot load the frozen champion",
        lambda: FrozenChampion.load(
            root,
            ledger=ExperimentLedger(root / "ledger"),
            scope_key=scope_key,
            config=config,
        ),
    )
    bundle = _guard(
        "Cannot load the serving bundle",
        lambda: load_serving_bundle(root, scope_key=champion.lock.scope_key),
    )

    manifest = bundle.manifest
    table = Table(title="Published serving bundle", show_header=False, box=None)
    table.add_row("Strategy", str(bundle.strategy))
    table.add_row("Model family", str(manifest.model_family))
    table.add_row("Champion lock", manifest.champion_lock_fingerprint[:16])
    table.add_row("Fusion selection", manifest.fusion_selection_fingerprint[:16])
    table.add_row(
        "Stacked state",
        (
            "not required by this strategy"
            if manifest.stacked_state_fingerprint is None
            else manifest.stacked_state_fingerprint[:16]
        ),
    )
    table.add_row("Files", str(len(manifest.files)))
    _console.print(table)
    _console.print("Every seal, digest, and lineage binding verified.")


def _rule_evidence(path: Path, what: str) -> Any:
    """Read Phase 4 risk assessments into per-anchor rule evidence.

    Identical, field for field, to what ``ml evaluate`` reads: the reconstruction
    has to see the rule arm the selection saw, and a rule decision derived even
    slightly differently here would be a different upstream input -- which the
    fingerprint comparison would catch, but as a mismatch rather than as the bug
    it is.

    The rule side always comes from a *published* detection run, never from
    re-running the engine.
    """
    from password_attack_detector.detection.serialization import read_risk_assessments
    from password_attack_detector.ml.fusion import RuleEvidence

    if not path.exists():
        _fail(f"Input not found: {_display(path)}")

    assessments = _guard(f"Cannot read the {what}", lambda: read_risk_assessments(path))
    evidence: dict[str, Any] = {}
    for assessment in assessments:
        evidence[assessment.anchor_event_id] = RuleEvidence(
            flagged=assessment.fired_rule_count > 0,
            ordinal_risk_score=float(assessment.risk_score),
        )
    return evidence


def _rule_configuration_fingerprint(config_path: Path | None) -> str:
    """Return the fingerprint of the frozen Phase 4 configuration."""
    from password_attack_detector.detection.config import (
        DetectionConfig,
        load_detection_config,
    )

    loaded = _guard(
        "Cannot load the detection configuration",
        lambda: (
            DetectionConfig()
            if config_path is None
            else load_detection_config(config_path)
        ),
    )
    return str(loaded.fingerprint())


def _validation_predictions(root: Path, prediction_id: str) -> Any:
    """Return the binary rows of a validated validation-split publication.

    Validation is not optional and nothing is read past a failure: a
    reconstruction fed a publication whose manifest, checksums, or lineage do not
    verify would produce a stacker that reads exactly like a sound one.
    """
    from password_attack_detector.ml.enums import MLSplit
    from password_attack_detector.ml.prediction_manifest import (
        BINARY_PREDICTION_FILE,
        PREDICTION_MANIFEST_FILE,
        PREDICTIONS_DIR,
        PredictionManifest,
    )
    from password_attack_detector.ml.prediction_serialization import (
        read_binary_predictions,
    )
    from password_attack_detector.ml.prediction_validation import validate_publication

    directory = Path(root) / PREDICTIONS_DIR / prediction_id
    if not directory.is_dir():
        _fail("That prediction publication does not exist under this artifact root")

    outcome = _guard(
        "Cannot validate the prediction publication",
        lambda: validate_publication(directory),
    )
    if not outcome.passed:
        _err.print(
            f"[red]Refusing to read an invalid publication:[/red] "
            f"{', '.join(outcome.failures)}"
        )
        raise typer.Exit(code=1)

    def _read() -> Any:
        manifest = PredictionManifest.from_json(
            (directory / PREDICTION_MANIFEST_FILE).read_text(encoding="utf-8")
        )
        if str(manifest.scope) != str(MLSplit.VALIDATION):
            _fail(
                "The publication given to --validation-prediction does not score "
                "the validation split; a hybrid selected on any other split "
                "would not be selected on validation"
            )
        return read_binary_predictions(directory / BINARY_PREDICTION_FILE)

    return _guard("Cannot read the prediction publication", _read)


def _assert_no_serving_time_fit() -> None:
    """Fail at import if this CLI ever gains a way to be called from the API.

    The separation this milestone rests on is that materialization is an
    operator action and serving is not.  The serving package must therefore not
    be importable *from here* either -- a deployment command that could construct
    a runtime would be one keystroke away from a runtime that could materialize.
    """
    import sys

    if "password_attack_detector.api.services" in sys.modules and (
        "materialize_serving_bundle"
        in vars(sys.modules["password_attack_detector.api.services"])
    ):
        raise ValueError(
            "the serving runtime imported the materializer; a process that "
            "serves a hybrid must not be able to fit one"
        )


_assert_no_serving_time_fit()
