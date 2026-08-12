"""Machine-learning CLI command group.

Subcommands::

    password-attack-detector ml catalog          -- the versioned model catalog
    password-attack-detector ml audit-features   -- the eligibility and leakage audit
    password-attack-detector ml verify-manifest  -- check a published model artifact
    password-attack-detector ml train            -- fit, calibrate, threshold, publish
    password-attack-detector ml experiments      -- list the immutable run ledger
    password-attack-detector ml select           -- gate candidates on validation-B
    password-attack-detector ml freeze-champion  -- freeze what a test may evaluate
    password-attack-detector ml predict          -- score rows under the frozen champion
    password-attack-detector ml validate         -- check a published prediction artifact
    password-attack-detector ml profile          -- the aggregate shape of that output

Ten commands, and the absences are deliberate.  Test evaluation, fusion, system
comparison, explainability, and drift arrive in later milestones, and no
placeholder is registered for them: a command that exists but does nothing is
worse than one that is honestly absent, because ``--help`` would advertise a
capability the code does not have.

**No command here reads a label.**  ``ml predict`` takes no ``--labels`` option,
``ml validate`` computes no accuracy, and ``ml profile`` reports the
*distribution* of what a model said rather than whether it was right.  Producing
predictions for the test split is safe precisely because everything that could
be tuned was frozen before this milestone ran; measuring them is a separate,
later, once-only step.

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


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def _selection_config(config_path: Path | None) -> Any:
    """Load the ML configuration a selection is carried out under."""
    from password_attack_detector.ml.config import MLConfig, load_ml_config

    return _guard(
        "Cannot load the ML configuration",
        lambda: MLConfig() if config_path is None else load_ml_config(config_path),
    )


def _gate_table(outcome: Any) -> Table:
    """Return the per-candidate gate table for one selection."""
    table = Table(title=f"Gates — {outcome.task}")
    table.add_column("Model")
    table.add_column("Gate")
    table.add_column("Status")
    table.add_column("Observed", justify="right")
    table.add_column("Required", justify="right")
    table.add_column("Support", justify="right")
    table.add_column("Reason")
    for candidate in outcome.results:
        for gate in candidate.gates:
            colour = {
                "pass": "green",
                "fail": "red",
                "inconclusive": "yellow",
            }[str(gate.status)]
            support = (
                "-"
                if gate.support_observed is None
                else f"{gate.support_observed:,}"
                + (
                    ""
                    if gate.support_required is None
                    else f"/{gate.support_required:,}"
                )
            )
            table.add_row(
                candidate.catalog_model_id,
                gate.gate_id,
                f"[{colour}]{gate.status}[/{colour}]",
                "unavailable" if gate.observed is None else f"{gate.observed:g}",
                "-" if gate.required is None else f"{gate.required:g}",
                support,
                gate.reason,
            )
    return table


def _outcome_table(outcome: Any) -> Table:
    """Return the per-candidate verdict table for one selection."""
    table = Table(title=f"Candidates — {outcome.task}")
    table.add_column("Model")
    table.add_column("Run")
    table.add_column("Eligibility")
    table.add_column("Blocking gates")
    table.add_column("Rank", justify="right")
    for candidate in outcome.results:
        position = (
            outcome.ranking.index(candidate.run_id) + 1
            if candidate.run_id in outcome.ranking
            else None
        )
        colour = "green" if str(candidate.status) == "eligible" else "yellow"
        table.add_row(
            candidate.catalog_model_id,
            candidate.run_id[:8],
            f"[{colour}]{candidate.status}[/{colour}]",
            ", ".join(candidate.blocking_gates) or "-",
            "-" if position is None else str(position),
        )
    return table


@ml_app.command()
def select(
    output_root: Annotated[
        Path | None,
        typer.Option(
            "--output-root",
            "-o",
            help="Root holding the published runs and the ledger.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="ML YAML configuration file.")
    ] = None,
    reports_dir: Annotated[
        Path | None,
        typer.Option("--reports-dir", help="Directory for the gate reports."),
    ] = None,
) -> None:
    """Select a champion from published training runs, on validation-B only.

    Reads immutable Milestone 6 run artifacts and the experiment ledger. It
    takes no feature, label, or split path, opens no Parquet table, and refits
    nothing: every number it compares was measured and frozen before this
    command ran. **The test split and the novel-anomaly holdout are read by
    nothing here, and there is no option that would let them be.**

    Every candidate is put through every mandatory gate, and each gate answers
    pass, fail, or inconclusive. A mandatory gate nobody could measure blocks
    promotion exactly as a failed one does -- a thin validation half is not a
    clean bill of health.

    The exit code carries the outcome:

    * ``0`` -- a champion was selected, and ``ml freeze-champion`` may proceed;
    * ``2`` -- no candidate cleared the gates, or the question could not be
      resolved on the available support. Neither is an error in the command, so
      neither is exit ``1``; both are findings, and both are recorded.

    Output is model identifiers, gate verdicts, counts, and stable reason codes.
    No event identifier, campaign, row, coefficient, threshold value, or
    absolute path is printed, and no test figure exists to print.
    """
    import json

    from password_attack_detector.ml.enums import ChampionStatus
    from password_attack_detector.ml.ledger import ExperimentLedger
    from password_attack_detector.ml.selection import (
        load_candidate_evidence,
        publish_selection,
        select_binary_champion,
        select_category_head,
        selection_report,
        selection_report_markdown,
    )

    config = _selection_config(config_path)
    root = output_root or (_artifacts_root() / "ml")
    ledger = ExperimentLedger(root / "ledger")

    evidence = _guard(
        "Cannot read the published training runs",
        lambda: load_candidate_evidence(root, ledger=ledger),
    )
    if not evidence:
        _fail(
            "No published training runs were found. Run 'ml train' first: a "
            "selection over nothing is not a selection."
        )

    binary = _guard(
        "Cannot select a binary champion",
        lambda: select_binary_champion(evidence, config=config),
    )
    category = _guard(
        "Cannot select a category head",
        lambda: select_category_head(evidence, config=config),
    )

    def _publish(outcome: Any) -> Any:
        """Publish one selection, converting a known failure into a clean exit."""
        return _guard(
            f"Cannot publish the {outcome.task} selection",
            lambda: publish_selection(outcome, root=root, ledger=ledger),
        )

    for outcome in (binary, category):
        _publish(outcome)

    target = reports_dir or Path("reports")
    target.mkdir(parents=True, exist_ok=True)
    (target / "ml_gates.json").write_text(
        json.dumps(
            {
                str(outcome.task): selection_report(outcome)
                for outcome in (binary, category)
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (target / "ml_gates.md").write_text(
        "\n".join(selection_report_markdown(outcome) for outcome in (binary, category)),
        encoding="utf-8",
    )

    _console.print(_outcome_table(binary))
    _console.print(_gate_table(binary))
    _console.print(_outcome_table(category))

    summary = Table(title="Selection", show_header=False, box=None)
    summary.add_row("Binary outcome", str(binary.status))
    summary.add_row("Binary selection", binary.record.record_id[:8])
    summary.add_row(
        "Reference baseline",
        "unavailable"
        if binary.reference_run_id is None
        else binary.reference_run_id[:8],
    )
    summary.add_row("Category outcome", str(category.status))
    summary.add_row("Category selection", category.record.record_id[:8])
    _console.print(summary)

    _console.print(f"Wrote {_display(target / 'ml_gates.json')}")
    _console.print(f"Wrote {_display(target / 'ml_gates.md')}")
    _console.print(
        "[dim]Validation-only. The test split and the novel-anomaly holdout "
        "were not read, and nothing here describes performance on unseen "
        "data.[/dim]"
    )

    if binary.status is not ChampionStatus.ELIGIBLE:
        _err.print(
            f"[yellow]No champion selected:[/yellow] {binary.status}. The "
            f"reference baseline is not a fallback -- it is the comparator every "
            f"candidate is measured against, and it is never promoted."
        )
        raise typer.Exit(code=2)

    _console.print(
        f"[green]Selected[/green] {binary.record.selected_run_id[:8]} "
        f"as binary champion candidate"
    )


# ---------------------------------------------------------------------------
# freeze-champion
# ---------------------------------------------------------------------------


@ml_app.command("freeze-champion")
def freeze_champion_command(
    output_root: Annotated[
        Path | None,
        typer.Option(
            "--output-root",
            "-o",
            help="Root holding the published runs, selections, and the ledger.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="ML YAML configuration file.")
    ] = None,
    selection_id: Annotated[
        str | None,
        typer.Option(
            "--selection",
            help=(
                "Validation-selection record to freeze. Defaults to the single "
                "eligible binary selection when there is exactly one."
            ),
        ),
    ] = None,
) -> None:
    """Freeze the champion an eligible validation selection chose.

    Re-reads and re-verifies everything the selection relied on -- the model
    artifact through the Milestone 4 verifier, every fingerprint against the
    artifact it names, and the candidate universe against the ledger -- before
    writing a lock. Trusting the selection record would make the lock a copy of
    a claim rather than a check of one.

    **There is no force option.** Freezing refuses when the selection found no
    champion, when it could not be resolved, when the chosen candidate did not
    clear a mandatory gate, when the reference baseline or an ineligible family
    is named, when a required calibrator or operating point is missing, when
    fingerprints disagree, or when the installed scikit-learn lies outside the
    reviewed range. Each of those is a state in which promoting would assert
    something nobody established.

    No test split is read, and no test evaluation is produced. The lock is what
    a later evaluation will be permitted to run; producing that evaluation is a
    separate, later, once-only step.
    """
    from password_attack_detector.ml.champion import (
        CHAMPION_LOCK_FILE,
        freeze_champion,
    )
    from password_attack_detector.ml.enums import ChampionStatus, MLTask
    from password_attack_detector.ml.ledger import ExperimentLedger
    from password_attack_detector.ml.selection import load_candidate_evidence

    config = _selection_config(config_path)
    root = output_root or (_artifacts_root() / "ml")
    ledger = ExperimentLedger(root / "ledger")

    selections = _guard(
        "Cannot read the experiment ledger", lambda: ledger.validation_selections()
    )
    binary = [item for item in selections if item.task is MLTask.BINARY_MALICIOUS]
    if selection_id is not None:
        chosen = [item for item in binary if item.record_id == selection_id]
        if not chosen:
            _fail("No binary validation selection with that identifier is recorded")
        selection = chosen[0]
    else:
        eligible = [item for item in binary if item.status is ChampionStatus.ELIGIBLE]
        if not eligible:
            _err.print(
                "[yellow]Nothing to freeze:[/yellow] no eligible binary "
                "selection is recorded. Run 'ml select' first, and if it found "
                "no champion, that is the finding."
            )
            raise typer.Exit(code=2)
        if len(eligible) > 1:
            _fail(
                "Several eligible binary selections are recorded; name one with "
                "--selection. Freezing whichever came first would make the "
                "champion depend on ledger enumeration order"
            )
        selection = eligible[0]

    category = next(
        (
            item
            for item in selections
            if item.task is MLTask.ATTACK_CATEGORY
            and item.status is ChampionStatus.ELIGIBLE
        ),
        None,
    )

    evidence = _guard(
        "Cannot read the published training runs",
        lambda: {
            item.run_id: item for item in load_candidate_evidence(root, ledger=ledger)
        },
    )
    publication = _guard(
        "Cannot freeze the champion",
        lambda: freeze_champion(
            selection,
            evidence=evidence,
            config=config,
            root=root,
            ledger=ledger,
            category=category,
        ),
    )

    table = Table(title="Frozen champion", show_header=False, box=None)
    table.add_row("Model", publication.catalog_model_id)
    table.add_row("Scope", publication.scope_key[:16])
    table.add_row("Lock fingerprint", publication.lock_fingerprint[:16])
    table.add_row("Selection", selection.record_id[:8])
    table.add_row("Freeze record", publication.record_id[:8])
    table.add_row(
        "Category head", "frozen" if category is not None else "none selected"
    )
    table.add_row("Newly written", _yes_no(publication.created))
    _console.print(table)
    _console.print(
        f"Lock at "
        f"{_display(root / 'champion' / publication.scope_key / CHAMPION_LOCK_FILE)}"
    )
    _console.print(
        "[dim]The lock names what a later evaluation may run. No test split has "
        "been read, and no test evaluation exists.[/dim]"
    )


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def _inference_inputs(
    *,
    features_path: Path,
    splits_path: Path,
    feature_manifest: Path,
    allowlist_path: Path,
    config_path: Path | None,
    feature_config_path: Path | None,
    scope: str,
) -> Any:
    """Resolve the feature contract and load the rows one prediction will score.

    Every path here is feature-side. There is no label parameter, and there is
    nowhere in the call below for one to go: the loader this composes takes
    exactly two tables, and neither is ground truth.
    """
    import json

    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.config import (
        FeatureConfig,
        load_feature_config,
    )
    from password_attack_detector.ml.config import MLConfig, load_ml_config
    from password_attack_detector.ml.dataset import load_inference_dataset
    from password_attack_detector.ml.enums import MLSplit
    from password_attack_detector.ml.features import (
        load_feature_allowlist,
        resolve_eligible_features,
    )

    for path in (features_path, splits_path, feature_manifest, allowlist_path):
        if not path.exists():
            _fail(f"Input not found: {_display(path)}")

    try:
        requested = MLSplit(scope)
    except ValueError:
        _fail(
            f"Unknown split {scope!r}; choose one of "
            f"{[str(item) for item in MLSplit if item is not MLSplit.EXCLUDED]}"
        )
    if requested is MLSplit.EXCLUDED:
        _fail(
            "Excluded rows are excluded from every stage of this layer, "
            "prediction included"
        )

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
        "Cannot assemble the inference input",
        lambda: load_inference_dataset(
            features_path=features_path,
            splits_path=splits_path,
            eligible=eligible,
            scope=requested,
            feature_catalog_fingerprint=catalog.fingerprint(),
        ),
    )
    return (config, catalog, allowlist, eligible, manifest, dataset)


@ml_app.command()
def predict(
    features_path: Annotated[
        Path, typer.Option("--features", help="Phase 3 feature snapshots Parquet file.")
    ],
    splits_path: Annotated[
        Path, typer.Option("--splits", help="Phase 3 feature splits Parquet file.")
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
    split: Annotated[
        str,
        typer.Option(
            "--split",
            help="Which split to score: train, validation, test, or "
            "novel_anomaly_holdout.",
        ),
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
            help="Root holding the frozen champion, the runs, and the ledger.",
        ),
    ] = None,
    scope_key: Annotated[
        str | None,
        typer.Option(
            "--scope-key",
            help=(
                "Which frozen champion scope to predict under. Required when "
                "more than one champion is frozen."
            ),
        ),
    ] = None,
    anomaly_run: Annotated[
        str | None,
        typer.Option(
            "--anomaly-run",
            help=(
                "Publish the named experimental anomaly run's scores in a "
                "separate artifact. Never read from champion.lock, and never "
                "able to change a supervised decision."
            ),
        ),
    ] = None,
    include_category: Annotated[
        bool,
        typer.Option(
            "--category/--no-category",
            help=(
                "Publish the frozen category head's assignments when one was "
                "frozen. A head that was not frozen is never fabricated."
            ),
        ),
    ] = True,
) -> None:
    """Score a split under the frozen champion and publish the predictions.

    Verifies the champion lock in full before a single row is scored -- the
    freeze receipt, the selection, the run, the model artifact, the manifest
    bytes, the preprocessor, the calibrator, the operating point, the whole
    feature contract, and the dependency ranges -- and refuses when any of them
    disagrees. **There is no force option, and no way to name a model instead of
    a lock:** predictions attributable to whichever model happened to be in a
    directory are attributable to nothing.

    **This command reads no labels.** There is no ``--labels`` option, the
    inference loader takes no label table, and nothing here computes a metric.
    Scoring the test split is therefore safe: everything that could be tuned was
    frozen before this ran, so a test prediction changes nothing, and changing a
    test label cannot change a byte of what is published.

    Output is the prediction identity, the scope, counts, and the validation
    verdict. No anchor identifier, row, feature value, pseudonym, coefficient,
    or absolute path is printed, and no outcome metric exists to print.
    """
    from password_attack_detector.ml.ledger import ExperimentLedger
    from password_attack_detector.ml.prediction_manifest import PREDICTIONS_DIR
    from password_attack_detector.ml.prediction_publisher import publish_predictions
    from password_attack_detector.ml.predictions import (
        ExperimentalAnomalyRun,
        FrozenChampion,
        verify_inference_feature_contract,
    )

    config, catalog, allowlist, eligible, manifest, dataset = _inference_inputs(
        features_path=features_path,
        splits_path=splits_path,
        feature_manifest=feature_manifest,
        allowlist_path=allowlist_path,
        config_path=config_path,
        feature_config_path=feature_config_path,
        scope=split,
    )

    root = output_root or (_artifacts_root() / "ml")
    ledger = ExperimentLedger(root / "ledger")

    champion = _guard(
        "Cannot load the frozen champion",
        lambda: FrozenChampion.load(
            root, ledger=ledger, scope_key=scope_key, config=config
        ),
    )
    _guard(
        "Feature contract",
        lambda: verify_inference_feature_contract(
            champion.lock,
            feature_manifest=manifest,
            catalog_fingerprint=catalog.fingerprint(),
            allowlist_fingerprint=allowlist.fingerprint(),
            eligible_feature_list_fingerprint=eligible.fingerprint(),
            required_feature_schema_version=config.required_feature_schema_version,
            compatible_catalog_fingerprints=(
                allowlist.compatible_feature_catalog_fingerprints
            ),
        ),
    )
    probe = (
        None
        if anomaly_run is None
        else _guard(
            "Cannot load the experimental anomaly run",
            lambda: ExperimentalAnomalyRun.load(root, anomaly_run, ledger=ledger),
        )
    )

    publication = _guard(
        "Cannot publish the predictions",
        lambda: publish_predictions(
            champion=champion,
            dataset=dataset,
            root=root,
            probe=probe,
            include_category=include_category,
        ),
    )

    table = Table(title="Predictions", show_header=False, box=None)
    table.add_row("Prediction", publication.prediction_id)
    table.add_row("Scope", f"{publication.scope} ({publication.scope_role})")
    table.add_row("Rows scored", f"{publication.row_count:,}")
    table.add_row("Model", publication.catalog_model_id)
    table.add_row("Model id", publication.model_id[:8])
    table.add_row(
        "Category head",
        "absent"
        if publication.category_row_count is None
        else f"{publication.category_row_count:,} row(s)",
    )
    table.add_row(
        "Experimental anomaly",
        "absent"
        if publication.anomaly_row_count is None
        else f"{publication.anomaly_row_count:,} row(s)",
    )
    table.add_row("Newly written", _yes_no(publication.created))
    table.add_row("Validation", str(publication.validation_status))
    _console.print(table)
    _console.print(
        f"Published under {_display(root / PREDICTIONS_DIR)}/"
        f"{publication.prediction_id}"
    )
    _console.print(
        "[dim]Predictions only. No label was read, no metric was computed, and "
        "nothing here says whether these predictions are correct.[/dim]"
    )

    if publication.validation_status is not _pass_status():
        _err.print(
            f"[red]Published artifact failed validation:[/red] "
            f"{', '.join(publication.validation_failures)}"
        )
        raise typer.Exit(code=1)


def _pass_status() -> Any:
    """Return the aggregate status that means every mandatory check passed."""
    from password_attack_detector.ml.enums import AuditStatus

    return AuditStatus.PASS


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _publication_directory(root: Path, prediction_id: str | None) -> Path:
    """Return the publication to act on, or exit naming why there is none."""
    from password_attack_detector.ml.prediction_manifest import (
        PREDICTION_MANIFEST_FILE,
        PREDICTIONS_DIR,
    )

    predictions_root = root / PREDICTIONS_DIR
    if not predictions_root.is_dir():
        _fail(
            "No predictions have been published under this root. Run "
            "'ml predict' first."
        )
    published = sorted(
        item.name
        for item in predictions_root.iterdir()
        if item.is_dir() and (item / PREDICTION_MANIFEST_FILE).is_file()
    )
    if not published:
        _fail("No complete prediction publication is present under this root")
    if prediction_id is None:
        if len(published) > 1:
            _fail(
                "Several prediction publications are present; name one with "
                "--prediction. Acting on whichever came first would make the "
                "result depend on directory enumeration order"
            )
        prediction_id = published[0]
    elif prediction_id not in published:
        _fail("No prediction publication with that identifier is present")
    return predictions_root / prediction_id


@ml_app.command()
def validate(
    output_root: Annotated[
        Path | None,
        typer.Option(
            "--output-root",
            "-o",
            help="Root holding the published predictions.",
        ),
    ] = None,
    prediction_id: Annotated[
        str | None,
        typer.Option(
            "--prediction",
            help=(
                "Which published prediction to validate. Defaults to the single "
                "publication when there is exactly one."
            ),
        ),
    ] = None,
) -> None:
    """Validate a published prediction artifact, without opening a label.

    Distinct from ``ml verify-manifest``, which checks a published *model*
    directory. This checks a published *prediction* directory: the manifest, the
    declared file set, every checksum, the derived prediction identity, the
    pinned Arrow schemas, the canonical row order, and -- row by row -- that each
    stored decision follows from that row's own stored score and threshold under
    the frozen predicate.

    **It computes no metric.** There is no accuracy here, no precision, and no
    recall, because establishing any of them requires the labels this milestone
    never reads. What validation says is that the artifact is what it claims to
    be, not that the model behind it is any good.

    A skipped mandatory check is not a pass. Exits non-zero on any invalid or
    tampered artifact, with a stable check code naming what failed.
    """
    from password_attack_detector.ml.enums import AuditCheckStatus
    from password_attack_detector.ml.prediction_validation import validate_publication

    root = output_root or (_artifacts_root() / "ml")
    directory = _publication_directory(root, prediction_id)
    result = _guard(
        "Cannot validate the prediction publication",
        lambda: validate_publication(directory),
    )

    table = Table(title="Prediction validation")
    table.add_column("Code")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Detail")
    for check in result.checks:
        colour = {
            AuditCheckStatus.PASS: "green",
            AuditCheckStatus.FAIL: "red",
            AuditCheckStatus.SKIPPED: "yellow",
        }[check.status]
        table.add_row(
            check.code,
            check.name,
            f"[{colour}]{check.status}[/{colour}]",
            check.detail,
        )
    _console.print(table)

    counts = Table(title="Counts", show_header=False, box=None)
    counts.add_row(
        "Scope", "unavailable" if result.scope is None else str(result.scope)
    )
    counts.add_row("Binary rows", f"{result.binary_row_count:,}")
    counts.add_row(
        "Category rows",
        "absent"
        if result.category_row_count is None
        else f"{result.category_row_count:,}",
    )
    counts.add_row(
        "Anomaly rows",
        "absent"
        if result.anomaly_row_count is None
        else f"{result.anomaly_row_count:,}",
    )
    _console.print(counts)
    _console.print(
        "[dim]Structural validity is not predictive quality. No label was read "
        "and no performance figure was computed.[/dim]"
    )

    if not result.passed:
        _err.print(f"[red]Validation FAILED:[/red] {', '.join(result.failures)}")
        raise typer.Exit(code=1)
    _console.print("[green]Validation PASS[/green]")


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@ml_app.command()
def profile(
    output_root: Annotated[
        Path | None,
        typer.Option(
            "--output-root",
            "-o",
            help="Root holding the published predictions.",
        ),
    ] = None,
    prediction_id: Annotated[
        str | None,
        typer.Option(
            "--prediction",
            help="Which published prediction to profile.",
        ),
    ] = None,
    reports_dir: Annotated[
        Path | None,
        typer.Option("--reports-dir", help="Directory for the profile reports."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: text or markdown."),
    ] = "text",
) -> None:
    """Report the aggregate shape of a **valid** published prediction artifact.

    Validation runs first and nothing is written unless it passes. Profiling a
    publication whose manifest, checksums, lineage, or rows do not verify would
    produce a document indistinguishable from a description of a sound artifact,
    so a tampered publication gets a non-zero exit, the failing check codes, and
    no report at all -- and any report from an earlier successful run is left
    exactly as it was. That artifact is ``ml validate``'s subject.

    Once it passes, the report is rebuilt from the publication alone -- the rows,
    the manifest, and that validation pass -- so it is checkable rather than
    merely informative. Nothing is taken from the training data, and no label is
    read.

    Every figure describes the *distribution of what the model said*: how many
    rows were flagged, where the scores sat, how often the category head
    abstained among the rows it was actually asked about. None of it describes
    whether any of that was right. A quantity nothing could produce is rendered
    as unavailable rather than as zero.
    """
    import json

    from password_attack_detector.ml.prediction_manifest import (
        ANOMALY_PREDICTION_FILE,
        BINARY_PREDICTION_FILE,
        CATEGORY_PREDICTION_FILE,
        PREDICTION_MANIFEST_FILE,
        QUALITY_REPORT_JSON_FILE,
        QUALITY_REPORT_MD_FILE,
        PredictionManifest,
    )
    from password_attack_detector.ml.prediction_serialization import (
        read_anomaly_scores,
        read_binary_predictions,
        read_category_predictions,
    )
    from password_attack_detector.ml.prediction_validation import validate_publication
    from password_attack_detector.ml.quality import (
        build_quality_report,
        quality_report_to_markdown,
    )

    if output_format not in {"text", "markdown"}:
        _fail(f"Unknown format {output_format!r}; use 'text' or 'markdown'")

    root = output_root or (_artifacts_root() / "ml")
    directory = _publication_directory(root, prediction_id)

    # Validation first, and nothing is written until it passes. A profile is a
    # description of a publication, so describing one whose manifest, checksums,
    # lineage, or rows do not verify would produce a document that reads exactly
    # like a description of a sound artifact. A tampered publication is
    # ``ml validate``'s subject, not this command's.
    outcome = _guard(
        "Cannot validate the prediction publication",
        lambda: validate_publication(directory),
    )
    if not outcome.passed:
        _err.print(
            f"[red]Refusing to profile an invalid publication:[/red] "
            f"{', '.join(outcome.failures)}"
        )
        _err.print(
            "No profile was written. Run 'ml validate' for the full check "
            "listing; profiling is not a way past manifest, checksum, or "
            "lineage verification."
        )
        raise typer.Exit(code=1)

    def _rebuild() -> Any:
        manifest = PredictionManifest.from_json(
            (directory / PREDICTION_MANIFEST_FILE).read_text(encoding="utf-8")
        )
        declared = {item.logical_name for item in manifest.files}
        binary = read_binary_predictions(directory / BINARY_PREDICTION_FILE)
        category = (
            read_category_predictions(directory / CATEGORY_PREDICTION_FILE)
            if CATEGORY_PREDICTION_FILE in declared
            else None
        )
        anomaly = (
            read_anomaly_scores(directory / ANOMALY_PREDICTION_FILE)
            if ANOMALY_PREDICTION_FILE in declared
            else None
        )
        return build_quality_report(
            manifest=manifest,
            validation=outcome,
            binary=binary,
            category=category,
            anomaly=anomaly,
        )

    report = _guard("Cannot profile the prediction publication", _rebuild)
    rendered = quality_report_to_markdown(report)

    target = reports_dir or Path("reports")
    target.mkdir(parents=True, exist_ok=True)
    (target / QUALITY_REPORT_JSON_FILE).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (target / QUALITY_REPORT_MD_FILE).write_text(rendered, encoding="utf-8")

    if output_format == "markdown":
        _console.print(rendered)
    else:
        _console.print(_profile_table(report))
        if report.category is not None:
            _console.print(_category_table(report))
        if report.anomaly is not None:
            _console.print(_anomaly_table(report))

    _console.print(f"Wrote {_display(target / QUALITY_REPORT_JSON_FILE)}")
    _console.print(f"Wrote {_display(target / QUALITY_REPORT_MD_FILE)}")
    _console.print(
        "[dim]Distribution and artifact quality only. No label was read, so no "
        "accuracy, precision, recall, or calibration error is computable from "
        "this publication.[/dim]"
    )


def _unavailable(value: Any) -> str:
    """Render a number, or say it is unavailable rather than printing a zero."""
    if value is None:
        return "unavailable"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:g}"


def _profile_table(report: Any) -> Table:
    """Return the binary distribution table for a profile."""
    binary = report.binary
    table = Table(title="Binary predictions", show_header=False, box=None)
    table.add_row("Prediction", report.prediction_id)
    table.add_row("Scope", f"{report.scope} ({report.scope_role.role})")
    table.add_row("Rows scored", f"{binary.total_rows:,}")
    table.add_row("Flagged", f"{binary.flagged_count:,}")
    table.add_row("Flagged rate", _unavailable(binary.flagged_rate))
    table.add_row("Score kind", str(binary.score_kind))
    table.add_row("Decision threshold", f"{binary.decision_threshold:g}")
    table.add_row(
        "Decision score range",
        f"{_unavailable(binary.decision_score_minimum)} .. "
        f"{_unavailable(binary.decision_score_maximum)}",
    )
    table.add_row(
        "Calibrated probability",
        "available" if binary.calibrated_probability_available else "unavailable",
    )
    table.add_row("Probability mean", _unavailable(binary.probability_mean))
    table.add_row("Rows without a probability", f"{binary.null_probability_count:,}")
    table.add_row("Validation", str(report.validation_status))
    return table


def _category_table(report: Any) -> Table:
    """Return the category triage table for a profile.

    The two absences are shown separately and named: a row the binary head never
    flagged was not asked, and a row that abstained was.
    """
    category = report.category
    table = Table(
        title="Category triage (of binary-positive rows)",
        caption=(
            f"{category.applicable_row_count:,} of {category.binary_row_count:,} "
            f"row(s) were routed to triage"
        ),
    )
    table.add_column("Outcome")
    table.add_column("Rows", justify="right")
    for item in category.class_counts:
        table.add_row(item.class_name, f"{item.predicted_count:,}")
    table.add_row("unknown (asked, abstained)", f"{category.unknown_count:,}")
    table.add_row(
        "not applicable (binary did not flag)", f"{category.not_applicable_count:,}"
    )
    return table


def _anomaly_table(report: Any) -> Table:
    """Return the experimental anomaly table for a profile."""
    anomaly = report.anomaly
    table = Table(title="Experimental anomaly probe", show_header=False, box=None)
    table.add_row("Rows scored", f"{anomaly.total_rows:,}")
    table.add_row(
        "Anomaly score range",
        f"{_unavailable(anomaly.score_minimum)} .. "
        f"{_unavailable(anomaly.score_maximum)}",
    )
    table.add_row("Threshold", _unavailable(anomaly.anomaly_threshold))
    table.add_row("Flagged", _unavailable(anomaly.flagged_count))
    table.add_row("Experimental", "yes")
    table.add_row("Influences champion selection", "no")
    return table
