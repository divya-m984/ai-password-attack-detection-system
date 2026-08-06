"""Rule-based detection CLI command group.

Subcommands::

    password-attack-detector detection run             -- full pipeline
    password-attack-detector detection validate        -- validate artifacts
    password-attack-detector detection profile         -- aggregate quality report
    password-attack-detector detection evaluate        -- ground-truth metrics
    password-attack-detector detection verify-manifest -- verify artifact integrity
    password-attack-detector detection catalog         -- print the rule catalog

**No command prints an identifier.**  Not an event identifier, a detection or
alert identifier, an entity-scope value, an evidence row, a coordinate, a
secret, or an absolute path.  Summaries are counts, distributions, and
fingerprints; paths are rendered relative to the working directory where
possible and fall back to a bare file name otherwise.

``detection run`` has **no** ``--labels`` and no ``--splits`` option.  The
absence is the enforcement: ground truth cannot reach the engine, the scorer,
or the alert builder because there is no argument to carry it.  ``detection
evaluate`` is the only command that reads labels or split assignments at all.

Heavy imports live inside the command bodies so ``--help`` stays fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from password_attack_detector.exceptions import (
    ArtifactNotFoundError,
    ConfigurationError,
    DataValidationError,
    DetectionConfigurationError,
    ManifestVerificationError,
    RuleEvaluationError,
)

detection_app = typer.Typer(
    name="detection",
    help=(
        "Rule-based authentication attack detection: run, validate, profile, "
        "evaluate, verify, and inspect the rule catalog."
    ),
    no_args_is_help=True,
)

_console = Console()
_err = Console(stderr=True)

#: Exceptions whose message is safe to show the user.  Every one of these is
#: raised by this project with a message written to carry counts and column
#: names only.  Anything else is reported by type name, so an unexpected
#: internal message can never reach a terminal.
_REPORTABLE = (
    ArtifactNotFoundError,
    ConfigurationError,
    DataValidationError,
    DetectionConfigurationError,
    ManifestVerificationError,
    RuleEvaluationError,
)


def _display(path: Path) -> str:
    """Render a path relative to the working directory where possible.

    Falls back to the bare file name, so an absolute path under a personal
    home directory never reaches the terminal.
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

    An unexpected exception is reported by type name alone: its message was
    not written with disclosure in mind, and a run log is not the place to
    find out.
    """
    try:
        return call()
    except _REPORTABLE as exc:
        _fail(f"{action}: {exc}")
    except Exception as exc:
        _fail(f"{action} failed ({type(exc).__name__})")


def _load_config(config_path: Path | None) -> Any:
    """Load a detection configuration, or the built-in defaults."""
    from password_attack_detector.detection.config import (
        DetectionConfig,
        load_detection_config,
    )

    if config_path is None:
        return DetectionConfig()
    return load_detection_config(config_path)


def _feature_catalog(feature_config_path: Path | None) -> Any:
    """Build the Phase 3 feature catalog the rules resolve against."""
    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.config import (
        FeatureConfig,
        load_feature_config,
    )

    config = (
        FeatureConfig()
        if feature_config_path is None
        else load_feature_config(feature_config_path)
    )
    return build_catalog(config)


def _check_feature_manifest(manifest_path: Path | None, config: Any) -> str | None:
    """Verify the source feature manifest and return its content fingerprint.

    Raises:
        DataValidationError: if the manifest is unreadable or declares a
            feature schema this configuration does not accept.  Running rules
            against a schema they were not written for would produce findings
            nobody could interpret.
    """
    if manifest_path is None:
        return None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataValidationError(
            f"Cannot read the feature manifest ({type(exc).__name__})"
        ) from None

    declared = raw.get("feature_schema_version")
    required = config.required_feature_schema_version
    if declared is not None and declared != required:
        raise DataValidationError(
            f"The feature manifest declares schema version {declared!r}, which "
            f"is incompatible with the configured requirement {required!r}"
        )
    fingerprint = raw.get("content_fingerprint")
    return str(fingerprint) if fingerprint is not None else None


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------


@detection_app.command()
def catalog(
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: text or markdown."),
    ] = "text",
    output_path: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write to this file instead of stdout."),
    ] = None,
) -> None:
    """Print the versioned rule catalog.

    Rendered from the same ``RULE_CATALOG`` the engine executes, so the
    documentation cannot describe a rule the code does not run.  The output is
    metadata: identifiers, versions, families, categories, feature templates,
    declared thresholds, evidence codes, severities, correlation groups, and
    limitations.  No executable configuration and no source code is emitted.
    """
    from password_attack_detector.detection.catalog import (
        RULE_CATALOG,
        RULE_CATALOG_VERSION,
        catalog_to_markdown,
    )

    if output_format not in {"text", "markdown"}:
        _fail(f"Unknown format {output_format!r}; use 'text' or 'markdown'")

    if output_format == "markdown":
        rendered = catalog_to_markdown(RULE_CATALOG)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered, encoding="utf-8")
            _console.print(f"Wrote rule catalog to {_display(output_path)}")
            return
        _console.print(rendered)
        return

    table = Table(title=f"Rule catalog v{RULE_CATALOG_VERSION}")
    for column in ("Rule", "Ver", "Family", "Category", "Severity", "Group"):
        table.add_column(column)
    for spec in RULE_CATALOG.specs:
        table.add_row(
            spec.rule_id,
            spec.rule_version,
            str(spec.family),
            str(spec.attack_category),
            str(spec.default_severity),
            str(spec.correlation_group),
        )
    _console.print(table)

    for spec in RULE_CATALOG.specs:
        _console.print(f"\n[bold]{spec.rule_id}[/bold] -- {spec.display_name}")
        _console.print(f"  Required features: {', '.join(spec.required_features)}")
        if spec.optional_features:
            _console.print(f"  Optional features: {', '.join(spec.optional_features)}")
        if spec.parameters:
            thresholds = ", ".join(
                f"{parameter.name}={parameter.default}" for parameter in spec.parameters
            )
            _console.print(f"  Thresholds: {thresholds}")
        _console.print(
            f"  Evidence codes: "
            f"{', '.join(item.evidence_code for item in spec.evidence)}"
        )
        for limitation in spec.limitations:
            _console.print(f"  Limitation: {limitation}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(catalog_to_markdown(RULE_CATALOG), encoding="utf-8")
        _console.print(f"\nWrote rule catalog to {_display(output_path)}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@detection_app.command()
def run(
    features_path: Annotated[
        Path,
        typer.Option("--features", help="Phase 3 feature snapshot Parquet file."),
    ],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Detection YAML configuration file."),
    ] = None,
    feature_manifest: Annotated[
        Path | None,
        typer.Option("--feature-manifest", help="Source feature manifest JSON."),
    ] = None,
    feature_config: Annotated[
        Path | None,
        typer.Option("--feature-config", help="Feature YAML configuration file."),
    ] = None,
    entity_scope: Annotated[
        Path | None,
        typer.Option(
            "--entity-scope",
            help="Optional pseudonymous scope table for alert grouping.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Directory for detection artifacts."),
    ] = None,
    reports_dir: Annotated[
        Path | None,
        typer.Option("--reports-dir", help="Directory for the quality report."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite existing output.")
    ] = False,
) -> None:
    """Detect, score, group, validate, and publish.

    The stage order is forced by the data.  Rules read one feature snapshot
    each and nothing else; scoring needs every rule that fired on an anchor;
    grouping needs every scored anchor; validation needs all three tables; and
    publication needs a validated set to describe.

    There is no ``--labels`` and no ``--splits`` option, and this command never
    reads either.  Ground truth belongs to ``detection evaluate``.
    """
    from password_attack_detector.detection.alerts import (
        AlertBuilder,
        build_entity_scope_table,
    )
    from password_attack_detector.detection.engine import DetectionEngine
    from password_attack_detector.detection.manifest import build_detection_manifest
    from password_attack_detector.detection.quality import (
        generate_detection_quality_report,
        report_to_json,
        report_to_markdown,
    )
    from password_attack_detector.detection.scoring import RiskScorer
    from password_attack_detector.detection.serialization import (
        ALERTS_FILE,
        DETECTIONS_FILE,
        MANIFEST_FILE,
        QUALITY_JSON_FILE,
        QUALITY_MD_FILE,
        RISK_FILE,
        DetectionPublisher,
        compute_alert_fingerprint,
        compute_detection_fingerprint,
        compute_risk_fingerprint,
        read_entity_scope,
        read_feature_snapshot_rows,
        write_alerts,
        write_fired_detections,
        write_risk_assessments,
    )
    from password_attack_detector.detection.validation import DetectionValidator

    config = _guard(
        "Cannot load the detection configuration", lambda: _load_config(config_path)
    )
    source_fingerprint = _guard(
        "Feature manifest check",
        lambda: _check_feature_manifest(feature_manifest, config),
    )
    feature_catalog = _guard(
        "Cannot build the feature catalog", lambda: _feature_catalog(feature_config)
    )

    if not features_path.exists():
        _fail(f"Feature snapshots not found: {_display(features_path)}")
    rows = _guard(
        "Cannot read the feature snapshots",
        lambda: read_feature_snapshot_rows(features_path),
    )
    if not rows:
        _fail("The feature snapshot table is empty; there is nothing to detect on")

    # Input validation happens inside the engine, before any rule sees a row:
    # a prohibited ground-truth, split, campaign, scope, or model column aborts
    # here rather than quietly influencing a finding.
    engine = _guard(
        "Cannot prepare the detection rules",
        lambda: DetectionEngine(config, feature_catalog=feature_catalog),
    )
    engine_result = _guard("Detection", lambda: engine.run(rows))
    diagnostics = _guard("Detection", lambda: engine.run_diagnostic(rows))

    scoring_result = _guard(
        "Risk scoring", lambda: RiskScorer(config).score(diagnostics)
    )

    scope_records = None
    if entity_scope is not None:
        if not entity_scope.exists():
            _fail(f"Entity-scope table not found: {_display(entity_scope)}")
        scope_records = _guard(
            "Cannot read the entity-scope table",
            lambda: read_entity_scope(entity_scope),
        )
    scope_table = (
        None
        if scope_records is None
        else _guard(
            "Entity-scope validation",
            lambda: build_entity_scope_table(scope_records),
        )
    )

    detections = list(engine_result.fired_detections)
    assessments = list(scoring_result.assessments)
    alerting_result = _guard(
        "Alert construction",
        lambda: AlertBuilder(config).build(
            assessments, detections=detections, entity_scope=scope_table
        ),
    )
    alerts = list(alerting_result.alerts)

    validation = DetectionValidator(config).validate(detections, assessments, alerts)
    if not validation.passed:
        _print_findings(validation)
        _fail(
            f"Detection artifacts failed validation with "
            f"{len(validation.errors)} error(s); nothing was published"
        )

    detection_fingerprint = compute_detection_fingerprint(detections)
    risk_fingerprint = compute_risk_fingerprint(assessments)
    alert_fingerprint = compute_alert_fingerprint(alerts)

    report = generate_detection_quality_report(
        engine_result=engine_result,
        scoring_result=scoring_result,
        alerting_result=alerting_result,
        config=config,
        validation_result=validation,
        input_snapshot_count=len(rows),
        detection_fingerprint=detection_fingerprint,
        risk_fingerprint=risk_fingerprint,
        alert_fingerprint=alert_fingerprint,
        source_feature_manifest_fingerprint=source_fingerprint,
    )

    target = output_dir or (config.output_dir or Path("data/processed"))
    reports_target = reports_dir or (config.reports_dir or target)
    overwrite = force or config.overwrite

    import tempfile

    times = [item.anchor_event_time for item in assessments]
    with tempfile.TemporaryDirectory(prefix="_pad_det_manifest_") as staging_name:
        staging = Path(staging_name)
        write_fired_detections(detections, staging / DETECTIONS_FILE)
        write_risk_assessments(assessments, staging / RISK_FILE)
        write_alerts(alerts, staging / ALERTS_FILE)
        manifest = build_detection_manifest(
            staging_dir=staging,
            config=config,
            detection_fingerprint=detection_fingerprint,
            risk_fingerprint=risk_fingerprint,
            alert_fingerprint=alert_fingerprint,
            detection_row_count=len(detections),
            risk_row_count=len(assessments),
            alert_row_count=len(alerts),
            earliest_event_time=min(times) if times else None,
            latest_event_time=max(times) if times else None,
            alert_grouping_mode=str(alerting_result.stats.grouping_mode),
            entity_scope_present=scope_table is not None,
            validation_status=str(validation.status),
            validation_result=validation.to_dict(),
            source_feature_manifest_fingerprint=source_fingerprint,
            quality_report_status=str(validation.status),
        )

    published = _guard(
        "Publication",
        lambda: DetectionPublisher(target, overwrite=overwrite).publish(
            detections,
            assessments,
            alerts,
            manifest.to_dict(),
            reports={
                QUALITY_JSON_FILE: report_to_json(report),
                QUALITY_MD_FILE: report_to_markdown(report),
            },
        ),
    )

    if reports_target.resolve() != target.resolve():
        reports_target.mkdir(parents=True, exist_ok=True)
        (reports_target / QUALITY_JSON_FILE).write_text(
            report_to_json(report), encoding="utf-8"
        )
        (reports_target / QUALITY_MD_FILE).write_text(
            report_to_markdown(report), encoding="utf-8"
        )

    _print_run_summary(report, validation, published.directory, MANIFEST_FILE)


def _print_run_summary(
    report: Any, validation: Any, directory: Path, manifest_name: str
) -> None:
    """Print an aggregate run summary.

    Counts, distributions, and relative paths only -- never an anchor, a
    detection or alert identifier, a scope value, or an evidence row.
    """
    table = Table(title="Detection run")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Evaluated snapshots", f"{report.evaluated_snapshot_count:,}")
    table.add_row("Rule evaluations", f"{report.total_rule_evaluation_count:,}")
    table.add_row("Fired detections", f"{report.fired_count:,}")
    table.add_row("Risk assessments", f"{report.evaluated_snapshot_count:,}")
    table.add_row("Insufficient data", f"{report.insufficient_data_count:,}")
    table.add_row("Alerts", f"{report.alert_count:,}")
    table.add_row("Suppressed", f"{report.suppressed_total:,}")
    table.add_row("Grouping mode", report.grouping_mode)
    table.add_row("Validation", report.validation_status or "n/a")
    _console.print(table)

    severities = Table(title="Assessment severity")
    severities.add_column("Severity")
    severities.add_column("Count", justify="right")
    for name, count in report.severity_distribution.items():
        severities.add_row(name, f"{count:,}")
    _console.print(severities)

    _console.print(f"Artifacts written to {_display(directory)}")
    _console.print(f"Manifest: {_display(directory / manifest_name)}")
    if validation.warnings:
        _console.print(
            f"[yellow]{len(validation.warnings)} validation warning(s); "
            f"see the quality report[/yellow]"
        )


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _print_findings(result: Any) -> None:
    """Print validation findings as codes, columns, and counts.

    A finding's message is written to carry no identifier, and the model
    rejects one that does -- so this is safe to print verbatim.
    """
    if not result.errors and not result.warnings:
        _console.print("[green]No findings.[/green]")
        return
    table = Table(title="Validation findings")
    table.add_column("Level")
    table.add_column("Code")
    table.add_column("Column")
    table.add_column("Count", justify="right")
    table.add_column("Finding")
    for finding in result.errors:
        table.add_row(
            "[red]error[/red]",
            finding.code,
            finding.column or "-",
            f"{finding.count:,}",
            finding.message,
        )
    for finding in result.warnings:
        table.add_row(
            "[yellow]warning[/yellow]",
            finding.code,
            finding.column or "-",
            f"{finding.count:,}",
            finding.message,
        )
    _console.print(table)


@detection_app.command()
def validate(
    detections_path: Annotated[
        Path, typer.Argument(help="Fired rule detections Parquet file.")
    ],
    risk_path: Annotated[
        Path, typer.Option("--risk-assessments", help="Risk assessments Parquet file.")
    ],
    alerts_path: Annotated[
        Path, typer.Option("--alerts", help="Security alerts Parquet file.")
    ],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Detection YAML configuration file."),
    ] = None,
    entity_scope: Annotated[
        Path | None,
        typer.Option("--entity-scope", help="Optional entity-scope table to check."),
    ] = None,
) -> None:
    """Validate a published detection artifact set.

    Exits non-zero when validation finds an error.  Warnings are reported and
    do not fail the command: a rule that never fired is worth knowing about
    and is not a defect.
    """
    from password_attack_detector.detection.serialization import (
        read_alerts,
        read_entity_scope,
        read_fired_detections,
        read_risk_assessments,
        read_table_columns,
    )
    from password_attack_detector.detection.validation import DetectionValidator

    config = _guard(
        "Cannot load the detection configuration", lambda: _load_config(config_path)
    )
    for path in (detections_path, risk_path, alerts_path):
        if not path.exists():
            _fail(f"Artifact not found: {_display(path)}")

    detections = _guard(
        "Cannot read the detections", lambda: read_fired_detections(detections_path)
    )
    assessments = _guard(
        "Cannot read the risk assessments", lambda: read_risk_assessments(risk_path)
    )
    alerts = _guard("Cannot read the alerts", lambda: read_alerts(alerts_path))
    scope = (
        None
        if entity_scope is None
        else _guard(
            "Cannot read the entity-scope table",
            lambda: read_entity_scope(entity_scope),
        )
    )

    result = DetectionValidator(config).validate(
        list(detections),
        list(assessments),
        list(alerts),
        entity_scope=None if scope is None else list(scope),
        detection_columns=read_table_columns(detections_path),
        risk_columns=read_table_columns(risk_path),
        alert_columns=read_table_columns(alerts_path),
    )

    summary = Table(title="Detection validation")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Status", str(result.status))
    summary.add_row("Detections", f"{result.detection_row_count:,}")
    summary.add_row("Risk assessments", f"{result.risk_assessment_row_count:,}")
    summary.add_row("Alerts", f"{result.alert_row_count:,}")
    summary.add_row("Invalid values", f"{result.invalid_value_count:,}")
    summary.add_row("Prohibited columns", f"{result.prohibited_column_count:,}")
    summary.add_row("Relationship errors", f"{result.relationship_error_count:,}")
    _console.print(summary)
    _print_findings(result)

    if not result.passed:
        raise typer.Exit(code=1) from None


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@detection_app.command()
def profile(
    detections_path: Annotated[
        Path, typer.Argument(help="Fired rule detections Parquet file.")
    ],
    risk_path: Annotated[
        Path, typer.Option("--risk-assessments", help="Risk assessments Parquet file.")
    ],
    alerts_path: Annotated[
        Path, typer.Option("--alerts", help="Security alerts Parquet file.")
    ],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Detection YAML configuration file."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Directory for the reports."),
    ] = None,
) -> None:
    """Write aggregate JSON and Markdown quality reports.

    The reports define what their numbers mean: risk score and signal strength
    are ordinal magnitudes and **not probabilities**, aggregate risk is the
    arithmetic mean of an alert's grouped assessments and peak risk their
    maximum, and evidence is an indicator rather than causal proof.

    This command reads *published artifacts*, not a live run. Counters only a
    run observes -- evaluations that did not fire, disabled rules, suppression
    decisions, gate rejections -- are reported as **unavailable**, never as
    zero. Zero would assert those events did not happen; the artifacts simply
    do not record them. Use the report written by ``detection run`` for a
    fully populated one.
    """
    from password_attack_detector.detection.quality import (
        RECONSTRUCTED_WARNING_CODE,
        reconstruct_detection_quality_report,
        report_to_json,
        report_to_markdown,
    )
    from password_attack_detector.detection.serialization import (
        QUALITY_JSON_FILE,
        QUALITY_MD_FILE,
        compute_alert_fingerprint,
        compute_detection_fingerprint,
        compute_risk_fingerprint,
        read_alerts,
        read_fired_detections,
        read_risk_assessments,
    )
    from password_attack_detector.detection.validation import DetectionValidator

    config = _guard(
        "Cannot load the detection configuration", lambda: _load_config(config_path)
    )
    for path in (detections_path, risk_path, alerts_path):
        if not path.exists():
            _fail(f"Artifact not found: {_display(path)}")

    detections = list(
        _guard(
            "Cannot read the detections",
            lambda: read_fired_detections(detections_path),
        )
    )
    assessments = list(
        _guard(
            "Cannot read the risk assessments",
            lambda: read_risk_assessments(risk_path),
        )
    )
    alerts = list(_guard("Cannot read the alerts", lambda: read_alerts(alerts_path)))

    validation = DetectionValidator(config).validate(detections, assessments, alerts)
    report = reconstruct_detection_quality_report(
        detections=detections,
        assessments=assessments,
        alerts=alerts,
        config=config,
        validation_result=validation,
        detection_fingerprint=compute_detection_fingerprint(detections),
        risk_fingerprint=compute_risk_fingerprint(assessments),
        alert_fingerprint=compute_alert_fingerprint(alerts),
    )

    target = output_dir or Path("reports")
    target.mkdir(parents=True, exist_ok=True)
    (target / QUALITY_JSON_FILE).write_text(report_to_json(report), encoding="utf-8")
    (target / QUALITY_MD_FILE).write_text(report_to_markdown(report), encoding="utf-8")

    _console.print(f"Wrote {_display(target / QUALITY_JSON_FILE)}")
    _console.print(f"Wrote {_display(target / QUALITY_MD_FILE)}")
    _console.print(
        f"[yellow]{RECONSTRUCTED_WARNING_CODE}: {report.reconstruction_note}[/yellow]"
    )
    _console.print(
        f"[dim]Unavailable counters: {', '.join(report.unavailable_metrics)}[/dim]"
    )
    for term, meaning in sorted(report.definitions.items()):
        _console.print(f"[dim]{term}[/dim]: {meaning}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@detection_app.command()
def evaluate(
    detections_path: Annotated[
        Path, typer.Option("--detections", help="Fired rule detections Parquet file.")
    ],
    risk_path: Annotated[
        Path, typer.Option("--risk-assessments", help="Risk assessments Parquet file.")
    ],
    alerts_path: Annotated[
        Path, typer.Option("--alerts", help="Security alerts Parquet file.")
    ],
    labels_path: Annotated[
        Path, typer.Option("--labels", help="Phase 3 feature labels Parquet file.")
    ],
    splits_path: Annotated[
        Path, typer.Option("--splits", help="Phase 3 feature splits Parquet file.")
    ],
    campaign_labels: Annotated[
        Path | None,
        typer.Option(
            "--campaign-labels",
            help="Phase 2 label table, for campaign-level metrics only.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Detection YAML configuration file."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Directory for the reports."),
    ] = None,
) -> None:
    """Evaluate a detection run against synthetic ground truth.

    **The only command that reads labels or split assignments.**  Nothing here
    feeds back into detection: no threshold is tuned, and the report records
    ``threshold_optimization_performed: false`` so the absence is stated rather
    than assumed.

    Train, validation, and test metrics are reported separately, and the
    novel-anomaly holdout is reported on its own -- its purpose is to measure
    behaviour on anomalies no rule was written for, and folding it into the
    supervised categories would destroy that measurement.
    """
    import pyarrow.parquet as pq

    from password_attack_detector.detection.evaluation import (
        CampaignRecord,
        LabelRecord,
        SplitRecord,
        evaluate_detection_run,
        report_to_json,
        report_to_markdown,
    )
    from password_attack_detector.detection.serialization import (
        EVALUATION_JSON_FILE,
        EVALUATION_MD_FILE,
        read_alerts,
        read_fired_detections,
        read_risk_assessments,
    )

    config = _guard(
        "Cannot load the detection configuration", lambda: _load_config(config_path)
    )
    required = (detections_path, risk_path, alerts_path, labels_path, splits_path)
    for path in required:
        if not path.exists():
            _fail(f"Input not found: {_display(path)}")

    detections = list(
        _guard(
            "Cannot read the detections", lambda: read_fired_detections(detections_path)
        )
    )
    assessments = list(
        _guard(
            "Cannot read the risk assessments", lambda: read_risk_assessments(risk_path)
        )
    )
    alerts = list(_guard("Cannot read the alerts", lambda: read_alerts(alerts_path)))

    def _rows(path: Path, what: str) -> list[dict[str, Any]]:
        try:
            return list(pq.read_table(path).to_pylist())
        except Exception as exc:
            raise DataValidationError(
                f"Cannot read the {what} ({type(exc).__name__})"
            ) from None

    label_rows = _guard("Labels", lambda: _rows(labels_path, "label table"))
    split_rows = _guard("Splits", lambda: _rows(splits_path, "split table"))
    labels = [
        LabelRecord(
            event_id=str(row["event_id"]),
            attack_class=str(row.get("attack_class") or "normal"),
            malicious=bool(row.get("malicious")),
            supervised_training_eligible=bool(
                row.get("supervised_training_eligible", True)
            ),
        )
        for row in label_rows
    ]
    splits = [
        SplitRecord(
            event_id=str(row["event_id"]),
            split=str(row.get("split") or ""),
            exclusion_reason=row.get("exclusion_reason"),
        )
        for row in split_rows
    ]

    campaigns = None
    if campaign_labels is not None:
        if not campaign_labels.exists():
            _fail(f"Campaign labels not found: {_display(campaign_labels)}")
        campaign_rows = _guard(
            "Campaign labels", lambda: _rows(campaign_labels, "campaign label table")
        )
        campaigns = [
            CampaignRecord(
                event_id=str(row["event_id"]), campaign_id=str(row["campaign_id"])
            )
            for row in campaign_rows
            if row.get("campaign_id")
        ]

    report = evaluate_detection_run(
        detections=detections,
        assessments=assessments,
        alerts=alerts,
        labels=labels,
        splits=splits,
        campaigns=campaigns,
        configuration_fingerprint=config.fingerprint(),
    )

    target = output_dir or Path("reports")
    target.mkdir(parents=True, exist_ok=True)
    (target / EVALUATION_JSON_FILE).write_text(report_to_json(report), encoding="utf-8")
    (target / EVALUATION_MD_FILE).write_text(
        report_to_markdown(report), encoding="utf-8"
    )

    _console.print(f"[yellow]{report.synthetic_caveat}[/yellow]")
    table = Table(title="Synthetic evaluation (overall)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    overall = report.overall
    for label, value in (
        ("Events", f"{overall.event_count:,}"),
        ("Benign", f"{overall.benign_count:,}"),
        ("Malicious", f"{overall.malicious_count:,}"),
        ("False-positive rate", _metric(overall.false_positive_rate)),
        ("Detection rate", _metric(overall.detection_rate)),
        ("Macro F1", _metric(overall.macro_f1)),
        ("Alerts per 1,000 events", _metric(overall.alerts_per_thousand_events)),
        ("Duplicate-reduction ratio", _metric(overall.duplicate_reduction_ratio)),
        ("Threshold optimization", str(report.threshold_optimization_performed)),
    ):
        table.add_row(label, value)
    _console.print(table)

    if report.novel_holdout is not None:
        _console.print(
            f"Novel-anomaly holdout reported separately: "
            f"{report.novel_holdout.event_count:,} event(s)"
        )
    _console.print(f"Wrote {_display(target / EVALUATION_JSON_FILE)}")
    _console.print(f"Wrote {_display(target / EVALUATION_MD_FILE)}")


def _metric(value: float | None) -> str:
    """Render a metric, or ``unavailable`` when its denominator was empty."""
    return "unavailable" if value is None else f"{value:,.4f}"


# ---------------------------------------------------------------------------
# verify-manifest
# ---------------------------------------------------------------------------


@detection_app.command(name="verify-manifest")
def verify_manifest_cmd(
    target: Annotated[
        Path,
        typer.Argument(
            help="Detection output directory, or its detection_manifest.json."
        ),
    ],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Detection YAML configuration file."),
    ] = None,
) -> None:
    """Verify the integrity of a published detection artifact set.

    Delegates path safety, checksums, and the primary row count to the shared
    verifier, then adds the detection-specific checks: content fingerprints,
    artifact roles, cross-table relationships, and -- when a configuration is
    supplied -- that the run's configuration and catalog are the ones in force.
    """
    from password_attack_detector.detection.manifest import verify_detection_dataset
    from password_attack_detector.detection.serialization import MANIFEST_FILE

    directory = target.parent if target.name == MANIFEST_FILE else target
    if not directory.exists():
        _fail(f"Detection directory not found: {_display(directory)}")

    config = (
        None
        if config_path is None
        else _guard(
            "Cannot load the detection configuration", lambda: _load_config(config_path)
        )
    )
    result = verify_detection_dataset(directory, config=config)

    table = Table(title="Detection manifest verification")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Detail")
    for check in result.checks:
        verdict = "[green]pass[/green]" if check.passed else "[red]FAIL[/red]"
        table.add_row(check.name, verdict, check.message)
    _console.print(table)

    if not result.passed:
        raise typer.Exit(code=1) from None
