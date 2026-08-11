"""Machine-learning CLI command group.

Subcommands::

    password-attack-detector ml catalog          -- the versioned model catalog
    password-attack-detector ml audit-features   -- the eligibility and leakage audit
    password-attack-detector ml verify-manifest  -- check a published model artifact
    password-attack-detector ml train            -- fit, calibrate, threshold, publish
    password-attack-detector ml experiments      -- list the immutable run ledger

Five commands, and the absences are deliberate.  Champion selection, test
evaluation, prediction, fusion, and comparison arrive in later milestones, and
no placeholder is registered for them: a command that exists but does nothing is
worse than one that is honestly absent, because ``--help`` would advertise a
capability the code does not have.

**No command prints an identifier.**  Not an event identifier, a campaign
identifier, an entity pseudonym, a coordinate, a raw feature row, a secret, or
an absolute path.  Output is metadata and aggregate counts.  No executable
configuration and no source code is emitted, and no measured performance figure
appears anywhere -- these commands describe what *may* be fitted, never how well
anything did.

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

ml_app = typer.Typer(
    name="ml",
    help=(
        "Machine-learning detection layer: inspect the model catalog, audit "
        "feature eligibility, train and publish immutable experiment runs, "
        "list the run ledger, and verify a published model artifact. Models "
        "are fitted on Phase 3 feature snapshots and are reported alongside "
        "the rule engine, never in place of it. No champion is selected and "
        "no test split is read."
    ),
    no_args_is_help=True,
)

_console = Console()
_err = Console(stderr=True)

#: Exceptions whose message is safe to show the user.  Every one is raised by
#: this project with a message written to carry counts and declared names only.
#: Anything else is reported by type name, so an unexpected internal message
#: can never reach a terminal.
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


def _display(path: Path) -> str:
    """Render a path relative to the working directory where possible.

    Falls back to the bare file name, so an absolute path under a personal home
    directory never reaches the terminal.
    """
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _fail(message: str) -> None:
    """Print an error and exit non-zero, without chaining a traceback."""
    _err.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=1) from None


def _guard(action: str, call: Any) -> Any:
    """Run *call*, converting a known failure into a sanitized exit.

    An unexpected exception is reported by type name alone: its message was not
    written with disclosure in mind, and a run log is not the place to find out.
    """
    try:
        return call()
    except _REPORTABLE as exc:
        _fail(f"{action}: {exc}")
    except Exception as exc:
        _fail(f"{action} failed ({type(exc).__name__})")


def _yes_no(value: bool) -> str:
    """Render a boolean for a terminal table."""
    return "yes" if value else "no"


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------


@ml_app.command()
def catalog(
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: text or markdown."),
    ] = "text",
    output_path: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write to this file instead of stdout."),
    ] = None,
    emit_allowlist: Annotated[
        Path | None,
        typer.Option(
            "--emit-allowlist",
            help="Draft a reviewable ML feature allowlist at this path and exit.",
        ),
    ] = None,
    feature_config_path: Annotated[
        Path | None,
        typer.Option(
            "--feature-config",
            help="Phase 3 feature YAML used to build the catalog for the draft.",
        ),
    ] = None,
) -> None:
    """Print the versioned model catalog.

    Rendered from the same ``MODEL_CATALOG`` the training path will read, so
    the documentation cannot describe a model family the code does not declare.

    Catalog membership is not championship: a family listed here may be fitted
    and evaluated, but promotion additionally requires proven serializer and
    inference-adapter parity plus every validation gate. Nothing printed here
    is a measured result.

    ``--emit-allowlist`` drafts a machine-learning feature allowlist from the
    Phase 3 catalog. The result is a **starting point for review**, never a
    finished contract: the rationale it writes for each feature is derived from
    that feature's own classification, which is precisely the reasoning a review
    exists to challenge.
    """
    from password_attack_detector.ml.catalog import (
        MODEL_CATALOG,
        MODEL_CATALOG_VERSION,
        model_catalog_to_markdown,
    )

    if output_format not in {"text", "markdown"}:
        _fail(f"Unknown format {output_format!r}; use 'text' or 'markdown'")

    if emit_allowlist is not None:
        _emit_allowlist(emit_allowlist, feature_config_path)
        return

    if output_format == "markdown":
        rendered = model_catalog_to_markdown(MODEL_CATALOG)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered, encoding="utf-8")
            _console.print(f"Wrote model catalog to {_display(output_path)}")
            return
        _console.print(rendered)
        return

    table = Table(title=f"Model catalog v{MODEL_CATALOG_VERSION}")
    for column in (
        "Model",
        "Ver",
        "Family",
        "Tasks",
        "Champion",
        "Experimental",
        "Calibratable",
        "Serializer",
    ):
        table.add_column(column)
    for spec in MODEL_CATALOG.specs:
        table.add_row(
            spec.model_id,
            spec.model_version,
            str(spec.family),
            ", ".join(str(task) for task in spec.supported_tasks),
            _yes_no(spec.champion_eligible),
            _yes_no(spec.experimental),
            _yes_no(spec.calibration_compatible),
            spec.serializer_id,
        )
    _console.print(table)

    for spec in MODEL_CATALOG.specs:
        _console.print(f"\n[bold]{spec.model_id}[/bold] -- {spec.display_name}")
        _console.print(f"  Eligibility: {spec.eligibility_status}")
        _console.print(f"  Native score kind: {spec.native_score_kind}")
        _console.print(f"  Inference adapter: {spec.inference_adapter_id}")
        _console.print(
            f"  Determinism controls: {', '.join(spec.determinism_controls)}"
        )
        if spec.public_estimator_attributes:
            _console.print(
                f"  Public estimator attributes: "
                f"{', '.join(spec.public_estimator_attributes)}"
            )
        if spec.private_estimator_attributes:
            _console.print(
                f"  Private estimator attributes: "
                f"{', '.join(spec.private_estimator_attributes)}"
            )
        if spec.hyperparameters:
            declared = ", ".join(
                f"{parameter.name}={parameter.default}"
                for parameter in spec.hyperparameters
            )
            _console.print(f"  Hyperparameters: {declared}")
        for limitation in spec.limitations:
            _console.print(f"  Limitation: {limitation}")

    _console.print(
        "\nCatalog membership does not make a model champion. Promotion "
        "requires proven serializer and inference parity and every validation "
        "gate. A model score is a probability only after calibration."
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            model_catalog_to_markdown(MODEL_CATALOG), encoding="utf-8"
        )
        _console.print(f"\nWrote model catalog to {_display(output_path)}")


def _emit_allowlist(target: Path, feature_config_path: Path | None) -> None:
    """Draft a reviewable feature allowlist and write it to *target*."""
    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.config import (
        FeatureConfig,
        load_feature_config,
    )
    from password_attack_detector.ml.features import emit_allowlist_document

    config = _guard(
        "Cannot load the feature configuration",
        lambda: (
            FeatureConfig()
            if feature_config_path is None
            else load_feature_config(feature_config_path)
        ),
    )
    catalog = _guard("Cannot build the feature catalog", lambda: build_catalog(config))
    document = emit_allowlist_document(
        catalog,
        compatible_feature_catalog_fingerprints=[catalog.fingerprint()],
        admitted_in="draft",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    _console.print(f"Wrote a draft feature allowlist to {_display(target)}")
    _console.print(
        "[yellow]This is a draft, not a reviewed contract. Every rationale it "
        "wrote is derived from the feature's own classification. Read it, "
        "replace the rationales, decide what to defer, and commit it.[/yellow]"
    )


# ---------------------------------------------------------------------------
# audit-features
# ---------------------------------------------------------------------------


@ml_app.command(name="audit-features")
def audit_features(
    features_path: Annotated[
        Path, typer.Option("--features", help="Phase 3 feature snapshots Parquet file.")
    ],
    labels_path: Annotated[
        Path, typer.Option("--labels", help="Phase 3 feature labels Parquet file.")
    ],
    splits_path: Annotated[
        Path, typer.Option("--splits", help="Phase 3 feature splits Parquet file.")
    ],
    campaign_labels: Annotated[
        Path,
        typer.Option(
            "--campaign-labels",
            help="Phase 2 label table, required for campaign-grouped validation.",
        ),
    ],
    feature_manifest: Annotated[
        Path,
        typer.Option(
            "--feature-manifest",
            help="Phase 3 feature manifest, for the fingerprint provenance check.",
        ),
    ],
    allowlist_path: Annotated[
        Path,
        typer.Option("--allowlist", help="Reviewed ML feature allowlist YAML file."),
    ],
    config_path: Annotated[
        Path | None, typer.Option("--config", help="ML YAML configuration file.")
    ] = None,
    feature_config_path: Annotated[
        Path | None,
        typer.Option("--feature-config", help="Phase 3 feature YAML configuration."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Directory for the audit reports."),
    ] = None,
) -> None:
    """Audit ML feature eligibility and leakage over a published feature dataset.

    Loads every data source through ``ml.dataset``, which is the only module in
    this layer permitted to read ground truth, split assignments, or campaign
    metadata. Resolves the reviewed allowlist against the executable catalog,
    assembles the canonical split-scoped dataset, partitions validation at
    campaign-group boundaries, and runs every currently implementable check.

    ``--campaign-labels`` and ``--feature-manifest`` are **required**, not
    optional. Campaign identifiers are absent from the Phase 3 tables, and a
    partition without them would have to cut between rows; the manifest is what
    the fingerprint provenance check compares against. Omitting either would
    leave a mandatory check unevaluated, and an unevaluated check is not a
    passed check -- so the command refuses the input rather than reporting a
    smaller audit as a clean one.

    Exits zero only when every check passed. Output is aggregate counts and
    stable check names; no identifier, campaign, row, pseudonym, or absolute
    path is printed.
    """
    import json

    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.config import (
        FeatureConfig,
        load_feature_config,
    )
    from password_attack_detector.ml.config import MLConfig, load_ml_config
    from password_attack_detector.ml.dataset import load_ml_dataset
    from password_attack_detector.ml.eligibility import (
        ML_AUDIT_JSON_FILE,
        ML_AUDIT_MD_FILE,
        MLEligibilityAuditor,
        ml_audit_result_to_markdown,
    )
    from password_attack_detector.ml.enums import MLSplit
    from password_attack_detector.ml.features import (
        load_feature_allowlist,
        resolve_eligible_features,
    )
    from password_attack_detector.ml.partition import partition_validation

    required = (
        features_path,
        labels_path,
        splits_path,
        campaign_labels,
        feature_manifest,
        allowlist_path,
    )
    for path in required:
        if not path.exists():
            _fail(f"Input not found: {_display(path)}")

    config: MLConfig = _guard(
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

    def _manifest() -> dict[str, Any]:
        try:
            loaded = json.loads(feature_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataValidationError(
                f"Cannot read the feature manifest ({type(exc).__name__})"
            ) from None
        if not isinstance(loaded, dict):
            raise DataValidationError("The feature manifest is not a JSON object")
        return loaded

    manifest = _guard("Feature manifest", _manifest)

    dataset = _guard(
        "Cannot assemble the dataset",
        lambda: load_ml_dataset(
            features_path=features_path,
            labels_path=labels_path,
            splits_path=splits_path,
            campaign_labels_path=campaign_labels,
            eligible=eligible,
            feature_catalog_fingerprint=catalog.fingerprint(),
        ),
    )

    partition = _guard(
        "Cannot partition the validation split",
        lambda: partition_validation(
            dataset.for_split(MLSplit.VALIDATION),
            config=config.validation_partition,
            support=config.support,
            campaign_metadata_supplied=True,
        ),
    )

    result = _guard(
        "Cannot audit the dataset",
        lambda: MLEligibilityAuditor(
            catalog=catalog,
            allowlist=allowlist,
            eligible=eligible,
            config=config,
            feature_manifest=manifest,
            partition=partition,
        ).audit(dataset),
    )

    target = output_dir or Path("reports")
    target.mkdir(parents=True, exist_ok=True)
    (target / ML_AUDIT_JSON_FILE).write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (target / ML_AUDIT_MD_FILE).write_text(
        ml_audit_result_to_markdown(result), encoding="utf-8"
    )

    table = Table(title="ML eligibility audit")
    table.add_column("Check")
    table.add_column("Result")
    for check in result.checks:
        colour = "green" if check.passed else "red"
        verdict = "pass" if check.passed else str(check.status)
        table.add_row(check.name, f"[{colour}]{verdict}[/{colour}]")
    _console.print(table)

    counts = Table(title="Counts")
    counts.add_column("Quantity")
    counts.add_column("Value", justify="right")
    counts.add_row("Eligible features", f"{result.checked_feature_count:,}")
    counts.add_row("Rows", f"{result.checked_row_count:,}")
    for split, count in sorted(result.split_row_counts.items()):
        counts.add_row(f"Rows in {split}", f"{count:,}")
    counts.add_row("Validation-A rows", f"{partition.partition_a_row_count:,}")
    counts.add_row("Validation-B rows", f"{partition.partition_b_row_count:,}")
    counts.add_row(
        "Validation-A campaigns", f"{partition.partition_a_campaign_count:,}"
    )
    counts.add_row(
        "Validation-B campaigns", f"{partition.partition_b_campaign_count:,}"
    )
    _console.print(counts)

    _console.print(f"Wrote {_display(target / ML_AUDIT_JSON_FILE)}")
    _console.print(f"Wrote {_display(target / ML_AUDIT_MD_FILE)}")
    _console.print(
        f"Allowlist [bold]{result.allowlist_id}[/bold] "
        f"v{result.allowlist_version} "
        f"({result.allowlist_fingerprint[:16]}); eligible feature list "
        f"{result.eligible_feature_list_fingerprint[:16]}"
    )
    _console.print(
        "[dim]A skipped check is not a passed check. Passing says the feature "
        "contract and split discipline are sound; it says nothing about "
        "detection effectiveness, and no model has been fitted.[/dim]"
    )

    if not result.passed:
        _err.print(f"[red]Audit failed:[/red] {', '.join(result.failures)}")
        raise typer.Exit(code=1)

    _console.print("[green]Audit status: pass[/green]")


# ---------------------------------------------------------------------------
# verify-manifest
# ---------------------------------------------------------------------------


@ml_app.command("verify-manifest")
def verify_manifest(
    target: Annotated[
        Path,
        typer.Argument(
            help="Published model directory to verify.",
            exists=False,
            dir_okay=True,
            file_okay=False,
        ),
    ],
) -> None:
    """Verify a published model artifact's structure, integrity, and identity.

    Reads JSON and hashes bytes. It does not construct an estimator, unpickle
    anything, import a module named by the artifact, or execute any part of it:
    a model directory is untrusted data, and verifying it must be safe to do to
    a directory somebody else wrote.

    What is printed is identity and contract -- the derived model identifier,
    the family, the task, the schema and serializer versions, and a file count.
    Never a coefficient, never a tree value, never a threshold, never a training
    row, and never an absolute path. Exits non-zero on any failure, with a
    stable error code.
    """
    from password_attack_detector.ml.manifest import verify_model_artifact

    outcome = _guard(
        "Cannot verify the model artifact", lambda: verify_model_artifact(target)
    )

    table = Table(title="Model artifact", show_header=False, box=None)
    table.add_row("Directory", _display(target))
    table.add_row("Model id", outcome.model_id or "unavailable")
    table.add_row("Family", outcome.model_family or "unavailable")
    table.add_row("Task", outcome.task or "unavailable")
    table.add_row("ML schema version", outcome.ml_schema_version or "unavailable")
    table.add_row(
        "Manifest schema version", outcome.manifest_schema_version or "unavailable"
    )
    table.add_row("Serializer", outcome.serializer_id or "unavailable")
    table.add_row(
        "Serializer version",
        "unavailable"
        if outcome.serializer_version is None
        else str(outcome.serializer_version),
    )
    table.add_row("Files", f"{outcome.file_count:,}")
    table.add_row("Checks run", f"{outcome.checks_run:,}")
    _console.print(table)

    if not outcome.passed:
        _err.print(
            f"[red]Verification FAILED[/red] [{outcome.error_code}]: "
            f"{outcome.error_detail}"
        )
        raise typer.Exit(code=1)

    _console.print("[green]Verification PASS[/green]")
    _console.print(
        "[dim]Structural and integrity verification only. No calibrator has "
        "been fitted, no champion has been selected, and no performance figure "
        "is recorded in a model artifact.[/dim]"
    )


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def _load_training_inputs(
    *,
    features_path: Path,
    labels_path: Path,
    splits_path: Path,
    campaign_labels: Path,
    feature_manifest: Path,
    allowlist_path: Path,
    config_path: Path | None,
    feature_config_path: Path | None,
) -> Any:
    """Load, audit, and freeze everything one training run is carried out under.

    Every data source is read through ``ml.dataset``, the one module in this
    layer permitted to see ground truth. Nothing here joins, filters, or fits:
    the CLI is a composition root, and a training decision made inside it would
    be a decision no test of the library could reach.
    """
    import json

    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.config import (
        FeatureConfig,
        load_feature_config,
    )
    from password_attack_detector.ml.config import MLConfig, load_ml_config
    from password_attack_detector.ml.dataset import load_ml_dataset
    from password_attack_detector.ml.eligibility import MLEligibilityAuditor
    from password_attack_detector.ml.enums import MLSplit
    from password_attack_detector.ml.features import (
        load_feature_allowlist,
        resolve_eligible_features,
    )
    from password_attack_detector.ml.partition import partition_validation
    from password_attack_detector.ml.training import TrainingContext

    for path in (
        features_path,
        labels_path,
        splits_path,
        campaign_labels,
        feature_manifest,
        allowlist_path,
    ):
        if not path.exists():
            _fail(f"Input not found: {_display(path)}")

    config: MLConfig = _guard(
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

    def _manifest() -> dict[str, Any]:
        try:
            loaded = json.loads(feature_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataValidationError(
                f"Cannot read the feature manifest ({type(exc).__name__})"
            ) from None
        if not isinstance(loaded, dict):
            raise DataValidationError("The feature manifest is not a JSON object")
        return loaded

    manifest = _guard("Feature manifest", _manifest)
    dataset = _guard(
        "Cannot assemble the dataset",
        lambda: load_ml_dataset(
            features_path=features_path,
            labels_path=labels_path,
            splits_path=splits_path,
            campaign_labels_path=campaign_labels,
            eligible=eligible,
            feature_catalog_fingerprint=catalog.fingerprint(),
        ),
    )
    partition = _guard(
        "Cannot partition the validation split",
        lambda: partition_validation(
            dataset.for_split(MLSplit.VALIDATION),
            config=config.validation_partition,
            support=config.support,
            campaign_metadata_supplied=True,
        ),
    )

    # The audit runs before anything is fitted, and a failure stops the run.
    # Training on a dataset whose feature contract or split discipline is
    # unsound would produce artifacts nobody should use, and somebody would
    # eventually use them.
    audit = _guard(
        "Cannot audit the dataset",
        lambda: MLEligibilityAuditor(
            catalog=catalog,
            allowlist=allowlist,
            eligible=eligible,
            config=config,
            feature_manifest=manifest,
            partition=partition,
        ).audit(dataset),
    )
    if not audit.passed:
        _err.print(f"[red]Eligibility audit failed:[/red] {', '.join(audit.failures)}")
        _err.print(
            "[red]Training refused.[/red] A model fitted on a dataset that "
            "failed the audit would be an artifact nobody should use."
        )
        raise typer.Exit(code=1)

    return _guard(
        "Cannot prepare the training context",
        lambda: TrainingContext.prepare(
            dataset,
            config=config,
            eligible=eligible,
            feature_catalog=catalog,
            partition=partition,
            allowlist_fingerprint=allowlist.fingerprint(),
        ),
    )


@ml_app.command()
def train(
    features_path: Annotated[
        Path, typer.Option("--features", help="Phase 3 feature snapshots Parquet file.")
    ],
    labels_path: Annotated[
        Path, typer.Option("--labels", help="Phase 3 feature labels Parquet file.")
    ],
    splits_path: Annotated[
        Path, typer.Option("--splits", help="Phase 3 feature splits Parquet file.")
    ],
    campaign_labels: Annotated[
        Path,
        typer.Option(
            "--campaign-labels",
            help="Phase 2 label table, required for campaign-grouped validation.",
        ),
    ],
    feature_manifest: Annotated[
        Path,
        typer.Option(
            "--feature-manifest",
            help="Phase 3 feature manifest, for the fingerprint provenance check.",
        ),
    ],
    allowlist_path: Annotated[
        Path,
        typer.Option("--allowlist", help="Reviewed ML feature allowlist YAML file."),
    ],
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
            help="Root for run artifacts and the ledger. Defaults to artifacts/ml.",
        ),
    ] = None,
) -> None:
    """Train every configured candidate and publish an immutable run for each.

    Audits feature eligibility first and refuses to train on a dataset that
    fails it. Then fits the binary head, the known-malicious category head, and
    the experimental anomaly probe -- each on the rows its own task permits --
    calibrates on validation-A, measures that calibrator on validation-B, and
    selects operating points on validation-B.

    The test split and the novel-anomaly holdout are read by nothing here. No
    champion is selected, no ``champion.lock`` is written, and no test metric is
    produced: this command records what was run, not which run won.

    A candidate that cannot be trained is reported with a status and the run
    continues, because a candidate list that silently shrinks is a comparison
    nobody can audit. The command exits non-zero only when the orchestration
    itself is invalid -- a failed audit, an unusable validation partition, or a
    publication that could not be completed.

    Output is run identifiers, statuses, tasks, and model identifiers. No event
    identifier, campaign, row, coefficient, threshold value, or absolute path is
    printed.
    """
    from password_attack_detector.ml.experiments import publish_training_run
    from password_attack_detector.ml.ledger import ExperimentLedger
    from password_attack_detector.ml.training import train_all

    context = _load_training_inputs(
        features_path=features_path,
        labels_path=labels_path,
        splits_path=splits_path,
        campaign_labels=campaign_labels,
        feature_manifest=feature_manifest,
        allowlist_path=allowlist_path,
        config_path=config_path,
        feature_config_path=feature_config_path,
    )

    root = output_root or (_artifacts_root() / "ml")
    ledger = ExperimentLedger(root / "ledger")

    outcomes = _guard(
        "Cannot train the configured candidates", lambda: train_all(context)
    )

    def _publish(outcome: Any) -> Any:
        """Publish one run, converting a known failure into a sanitized exit."""
        return _guard(
            f"Cannot publish the {outcome.candidate.label} run",
            lambda: publish_training_run(
                outcome, context=context, root=root, ledger=ledger
            ),
        )

    publications = [_publish(outcome) for outcome in outcomes]

    table = Table(title="Training runs")
    table.add_column("Run")
    table.add_column("Model")
    table.add_column("Task")
    table.add_column("Status")
    table.add_column("Model id")
    for publication, outcome in zip(publications, outcomes, strict=True):
        colour = "green" if outcome.complete else "yellow"
        record_model_id = (
            "unavailable" if outcome.fitted is None else _model_identifier(outcome)
        )
        table.add_row(
            publication.run_id[:8],
            publication.catalog_model_id,
            str(publication.task),
            f"[{colour}]{publication.status}[/{colour}]",
            record_model_id,
        )
    _console.print(table)

    counts = Table(title="Support", show_header=False, box=None)
    counts.add_row("Validation-A rows", f"{context.partition.partition_a_row_count:,}")
    counts.add_row("Validation-B rows", f"{context.partition.partition_b_row_count:,}")
    counts.add_row(
        "Runs published", f"{sum(1 for item in publications if item.created):,}"
    )
    counts.add_row(
        "Runs already present",
        f"{sum(1 for item in publications if not item.created):,}",
    )
    _console.print(counts)

    _console.print(f"Wrote runs under {_display(root / 'runs')}")
    _console.print(f"Ledger at {_display(root / 'ledger')}")
    _console.print(
        "[dim]No champion has been selected and no test split has been read. "
        "A completed run means every artifact its task requires was published, "
        "not that the model is any good.[/dim]"
    )


def _artifacts_root() -> Path:
    """Return the project artifacts directory."""
    from password_attack_detector.paths import get_artifacts_dir

    return get_artifacts_dir()


def _model_identifier(outcome: Any) -> str:
    """Return the derived model identifier for a fitted outcome."""
    from password_attack_detector.ml.serialization import model_id_for

    fitted = outcome.fitted
    return model_id_for(
        fitted.content_fingerprint(), task=fitted.task, family=fitted.family
    )[:8]


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------


@ml_app.command()
def experiments(
    output_root: Annotated[
        Path | None,
        typer.Option(
            "--output-root",
            "-o",
            help="Root holding the ledger. Defaults to artifacts/ml.",
        ),
    ] = None,
    reconcile_runs: Annotated[
        bool,
        typer.Option(
            "--reconcile/--no-reconcile",
            help=(
                "Index any published run the ledger does not yet hold. Appends "
                "only; nothing stored is ever rewritten."
            ),
        ),
    ] = False,
) -> None:
    """List the immutable training runs the experiment ledger holds.

    Identity and status only: run identifier, record type, task, model
    identifier and family, run status, calibration method and outcome, and
    threshold outcome. **No metric of any kind is shown** -- not a validation
    score, and certainly not a test one -- because a listing that ranked runs
    would be a champion selection under another name.

    ``--reconcile`` appends a ledger record for any complete published run the
    ledger is missing, reading the record from the run directory it was
    published with. It is the recovery for a run that was promoted and then not
    indexed; it rewrites nothing and deletes nothing.
    """
    from password_attack_detector.ml.enums import ExperimentRecordType
    from password_attack_detector.ml.experiments import reconcile, summarize
    from password_attack_detector.ml.ledger import ExperimentLedger

    root = output_root or (_artifacts_root() / "ml")
    ledger = ExperimentLedger(root / "ledger")

    if reconcile_runs:
        appended = _guard(
            "Cannot reconcile the ledger", lambda: reconcile(root=root, ledger=ledger)
        )
        _console.print(f"Indexed {len(appended):,} previously unindexed run(s)")

    records = _guard(
        "Cannot read the experiment ledger", lambda: ledger.training_runs()
    )
    if not records:
        _console.print("The experiment ledger holds no training runs.")
        return

    # Two tables rather than one wide one. Status names are long by design --
    # ``insufficient_validation_support`` says exactly what happened -- and a
    # single table would have to truncate them on an ordinary terminal, which
    # is the one thing a status column must never do.
    summaries = [summarize(record) for record in records]

    identity = Table(
        title="Experiment ledger",
        caption=f"{len(summaries):,} immutable training_run record(s)",
    )
    identity.add_column("Run")
    identity.add_column("Model")
    identity.add_column("Task")
    identity.add_column("Status")
    identity.add_column("Flags")
    for summary in summaries:
        assert summary.record_type is ExperimentRecordType.TRAINING_RUN
        flags = [
            name
            for name, present in (
                ("reference", summary.reference_baseline),
                ("experimental", summary.experimental),
                ("eligible", summary.champion_eligible),
            )
            if present
        ]
        colour = "green" if summary.status.complete else "yellow"
        identity.add_row(
            summary.run_id[:8],
            summary.catalog_model_id,
            str(summary.task),
            f"[{colour}]{summary.status}[/{colour}]",
            ", ".join(flags) or "-",
        )
    _console.print(identity)

    operating = Table(title="Operating points")
    operating.add_column("Run")
    operating.add_column("Model")
    operating.add_column("Calibration")
    operating.add_column("Threshold")
    for summary in summaries:
        operating.add_row(
            summary.run_id[:8],
            summary.catalog_model_id,
            f"{summary.calibration_method}/{summary.calibration_status}",
            summary.threshold_status,
        )
    _console.print(operating)
    _console.print(
        "[dim]Training runs only. No champion has been selected, no test "
        "evaluation exists, and no figure here describes performance.[/dim]"
    )
