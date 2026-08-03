"""Feature-engineering CLI command group.

Subcommands:

    password-attack-detector features build           -- full pipeline
    password-attack-detector features split           -- split assignments only
    password-attack-detector features fit-baseline    -- fit and persist a baseline
    password-attack-detector features transform       -- apply an existing baseline
    password-attack-detector features audit-leakage   -- run the twelve checks
    password-attack-detector features validate        -- validate a feature table
    password-attack-detector features profile         -- aggregate quality report
    password-attack-detector features verify-manifest -- verify artifact integrity
    password-attack-detector features catalog         -- print the feature catalog

No command prints an event identifier, a pseudonym, a coordinate, or an
absolute path.  Baseline summaries read the metadata-only ``baseline.json``,
never the pseudonym-bearing tables beside it.

Heavy imports live inside the command bodies so ``--help`` stays fast.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from password_attack_detector.exceptions import (
    ArtifactNotFoundError,
    BaselineFitError,
    ConfigurationError,
    DataValidationError,
    FeatureComputationError,
    SplitConfigurationError,
)

features_app = typer.Typer(
    name="features",
    help=(
        "Point-in-time feature engineering: build, split, fit baselines, "
        "audit for leakage, validate, and profile."
    ),
    no_args_is_help=True,
)

_console = Console()
_err = Console(stderr=True)

#: Exceptions whose message is safe to show the user.  Anything else is
#: reported by type name only, so internals never leak into a terminal.
_REPORTABLE = (
    ArtifactNotFoundError,
    BaselineFitError,
    ConfigurationError,
    DataValidationError,
    FeatureComputationError,
    SplitConfigurationError,
)


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


def _load_config(config_path: Path | None) -> Any:
    from password_attack_detector.features.config import (
        FeatureConfig,
        load_feature_config,
    )

    if config_path is None:
        return FeatureConfig()
    return load_feature_config(config_path)


def _parse_boundary(value: str | None) -> datetime | None:
    if value is None:
        return None
    from datetime import UTC

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ConfigurationError(f"Boundary {value!r} must include a timezone offset")
    return parsed.astimezone(UTC)


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------


@features_app.command()
def catalog(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Feature YAML configuration file."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: text, json, or markdown."),
    ] = "text",
    output_path: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write to this file instead of stdout."),
    ] = None,
) -> None:
    """Print the feature catalog implied by a configuration."""
    import json as json_module

    from password_attack_detector.features.catalog import (
        build_catalog,
        catalog_to_markdown,
    )
    from password_attack_detector.features.config import FEATURE_SCHEMA_VERSION

    try:
        config = _load_config(config_path)
        built = build_catalog(config)
    except _REPORTABLE as exc:
        _fail(str(exc))
        return

    if output_format == "markdown":
        rendered = catalog_to_markdown(built)
    elif output_format == "json":
        rendered = json_module.dumps(
            {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "catalog_fingerprint": built.fingerprint(),
                "config_fingerprint": built.config_fingerprint,
                "feature_count": len(built),
                "features": [spec.model_dump(mode="json") for spec in built.specs],
            },
            indent=2,
            sort_keys=True,
        )
    elif output_format == "text":
        table = Table(title=f"Feature catalog ({len(built)} features)")
        table.add_column("Group")
        table.add_column("Features", justify="right")
        for group, count in sorted(built.group_counts().items()):
            table.add_row(group, str(count))
        _console.print(table)
        _console.print(f"Catalog fingerprint: {built.fingerprint()}")
        _console.print(f"Config fingerprint:  {built.config_fingerprint}")
        return
    else:
        _fail(f"Unknown format {output_format!r}; expected text, json, or markdown")
        return

    if output_path is None:
        _console.print(rendered)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        _console.print(f"[green]Wrote[/green] {_display(output_path)}")


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


@features_app.command()
def split(
    events_path: Annotated[
        Path, typer.Argument(help="Canonical authentication events Parquet file.")
    ],
    labels_path: Annotated[
        Path, typer.Option("--labels", help="Ground-truth labels Parquet file.")
    ],
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Feature YAML configuration file.")
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Directory for the split table."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite existing output.")
    ] = False,
) -> None:
    """Produce split assignments without computing any features."""
    import json as json_module

    from password_attack_detector.data.serialization import read_events_parquet
    from password_attack_detector.features.serialization import (
        SPLITS_FILE,
        read_ground_truth_labels,
        write_feature_splits,
    )
    from password_attack_detector.features.splitting import split_dataset

    try:
        config = _load_config(config_path)
        events = read_events_parquet(events_path)
        labels = read_ground_truth_labels(labels_path)
        result = split_dataset(events, labels, config)
    except _REPORTABLE as exc:
        _fail(str(exc))
        return
    except Exception as exc:
        _fail(f"Unexpected failure ({type(exc).__name__})")
        return

    _print_split_summary(result.manifest)

    if output_dir is None:
        return

    target = output_dir / SPLITS_FILE
    manifest_target = output_dir / "split_manifest.json"
    if not force and (target.exists() or manifest_target.exists()):
        _fail(f"Split output already exists in {_display(output_dir)}; use --force")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_feature_splits(result.assignments, target)
    manifest_target.write_text(
        json_module.dumps(result.manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _console.print(f"[green]Wrote[/green] {_display(target)}")
    _console.print(f"[green]Wrote[/green] {_display(manifest_target)}")


def _print_split_summary(manifest: Any) -> None:
    """Print aggregate split sizes.  Never prints an event identifier."""
    table = Table(title="Split assignment")
    table.add_column("Split")
    table.add_column("Events", justify="right")
    for name, count in sorted(manifest.split_counts.items()):
        table.add_row(name, f"{count:,}")
    _console.print(table)

    if manifest.exclusion_counts:
        reasons = Table(title="Exclusions")
        reasons.add_column("Reason")
        reasons.add_column("Events", justify="right")
        for reason, count in sorted(manifest.exclusion_counts.items()):
            reasons.add_row(reason, f"{count:,}")
        _console.print(reasons)


# ---------------------------------------------------------------------------
# fit-baseline
# ---------------------------------------------------------------------------


@features_app.command(name="fit-baseline")
def fit_baseline_cmd(
    events_path: Annotated[
        Path, typer.Argument(help="Canonical authentication events Parquet file.")
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Directory for the baseline.")
    ],
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Feature YAML configuration file.")
    ] = None,
    split_table: Annotated[
        Path | None,
        typer.Option("--split-table", help="Split assignments Parquet file."),
    ] = None,
    train_end: Annotated[
        str | None,
        typer.Option("--train-end", help="ISO-8601 end of the training interval."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing baseline.")
    ] = False,
) -> None:
    """Fit a behavioral baseline on approved training events only.

    Requires either an explicit split table or a training cutoff.  A baseline
    is never silently fitted on validation or test data.
    """
    import pyarrow.parquet as pq

    from password_attack_detector.data.serialization import read_events_parquet
    from password_attack_detector.features.baselines import BehavioralBaselineModel

    if (split_table is None) == (train_end is None):
        _fail(
            "Provide exactly one of --split-table or --train-end so the "
            "training set is explicit"
        )
        return

    try:
        config = _load_config(config_path)
        events = read_events_parquet(events_path)

        if split_table is not None:
            rows = pq.read_table(split_table).to_pylist()
            train_ids = {r["event_id"] for r in rows if r["split"] == "train"}
            training = [e for e in events if str(e.event_id) in train_ids]
        else:
            cutoff = _parse_boundary(train_end)
            assert cutoff is not None
            training = [e for e in events if e.event_time < cutoff]

        if not training:
            _fail("The training selection is empty; nothing to fit")
            return

        interval = (
            min(e.event_time for e in training),
            max(e.event_time for e in training),
        )
        model = BehavioralBaselineModel(config.baseline).fit(
            training,
            permitted_event_ids=frozenset(e.event_id for e in training),
            interval=interval,
        )
        model.save(output_dir, overwrite=force)
    except _REPORTABLE as exc:
        _fail(str(exc))
        return
    except Exception as exc:
        _fail(f"Unexpected failure ({type(exc).__name__})")
        return

    _print_baseline_summary(model.artifact)
    _console.print(f"[green]Wrote baseline to[/green] {_display(output_dir)}")


def _print_baseline_summary(artifact: Any) -> None:
    """Print baseline metadata only -- never the pseudonym-bearing tables."""
    table = Table(title="Fitted baseline")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Users", f"{artifact.user_count:,}")
    table.add_row("Sources", f"{artifact.source_count:,}")
    table.add_row("Events consumed", f"{artifact.total_fit_events:,}")
    table.add_row("Truncated sets", f"{artifact.truncated_set_count:,}")
    table.add_row("Content fingerprint", artifact.content_fingerprint)
    _console.print(table)


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------


@features_app.command()
def transform(
    events_path: Annotated[
        Path, typer.Argument(help="Canonical authentication events Parquet file.")
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Directory for the snapshots.")
    ],
    baseline_dir: Annotated[
        Path | None,
        typer.Option("--baseline", help="Directory holding a fitted baseline."),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Feature YAML configuration file.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite existing output.")
    ] = False,
) -> None:
    """Compute feature snapshots using an existing baseline.

    No split is produced and no baseline is fitted, so this is the shape a
    production scoring path would take.
    """
    from password_attack_detector.data.serialization import read_events_parquet
    from password_attack_detector.features.baselines import BehavioralBaselineModel
    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.engine import FeatureEngine
    from password_attack_detector.features.serialization import (
        SNAPSHOTS_FILE,
        write_feature_snapshots,
    )

    try:
        config = _load_config(config_path)
        events = read_events_parquet(events_path)
        model = (
            None
            if baseline_dir is None
            else BehavioralBaselineModel.load(baseline_dir, config.baseline)
        )
        frame = FeatureEngine(config, build_catalog(config), baseline=model).run(events)
    except _REPORTABLE as exc:
        _fail(str(exc))
        return
    except Exception as exc:
        _fail(f"Unexpected failure ({type(exc).__name__})")
        return

    target = output_dir / SNAPSHOTS_FILE
    if target.exists() and not force:
        _fail(f"{_display(target)} already exists; use --force")
        return

    write_feature_snapshots(frame, target)
    _console.print(
        f"[green]Wrote[/green] {_display(target)} "
        f"({len(frame.rows):,} rows x {len(frame.catalog)} features)"
    )


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


@features_app.command()
def build(
    events_path: Annotated[
        Path, typer.Argument(help="Canonical authentication events Parquet file.")
    ],
    labels_path: Annotated[
        Path, typer.Option("--labels", help="Ground-truth labels Parquet file.")
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for the feature dataset."),
    ],
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Feature YAML configuration file.")
    ] = None,
    baseline_dir: Annotated[
        Path | None,
        typer.Option("--baseline", help="Reuse this baseline instead of fitting one."),
    ] = None,
    reports_dir: Annotated[
        Path | None,
        typer.Option("--reports-dir", help="Directory for quality and audit reports."),
    ] = None,
    skip_audit: Annotated[
        bool,
        typer.Option("--skip-audit", help="Skip the leakage audit (not recommended)."),
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite existing output.")
    ] = False,
) -> None:
    """Run the full pipeline: split, fit, transform, audit, validate, publish.

    The order is forced by the data, not by preference.  The split depends only
    on timestamps and labels; the baseline depends on the split; the engine
    depends on the baseline.  The engine then runs over the **entire** event
    stream regardless of split -- isolation comes from purging events whose
    lookback windows cross a boundary, never from resetting engine state, which
    would manufacture cold-start artifacts that do not occur in production.
    """
    import json as json_module
    import tempfile

    from password_attack_detector.data.serialization import (
        compute_events_fingerprint,
        read_events_parquet,
    )
    from password_attack_detector.features.baselines import BehavioralBaselineModel
    from password_attack_detector.features.catalog import (
        ANCHOR_EVENT_ID,
        build_catalog,
    )
    from password_attack_detector.features.engine import (
        FeatureEngine,
        sort_events_canonically,
    )
    from password_attack_detector.features.leakage import (
        LeakageAuditor,
        audit_result_to_markdown,
    )
    from password_attack_detector.features.manifest import build_feature_manifest
    from password_attack_detector.features.quality import (
        generate_feature_quality_report,
        report_to_json,
        report_to_markdown,
    )
    from password_attack_detector.features.serialization import (
        FeaturePublisher,
        compute_features_fingerprint,
        labels_for_events,
        read_ground_truth_labels,
        write_feature_labels,
        write_feature_snapshots,
        write_feature_splits,
    )
    from password_attack_detector.features.splitting import (
        SplitLabel,
        split_dataset,
    )
    from password_attack_detector.features.validation import FeatureValidator

    try:
        config = _load_config(config_path)
        built_catalog = build_catalog(config)

        events = sort_events_canonically(read_events_parquet(events_path))
        labels = read_ground_truth_labels(labels_path)
        if not events:
            _fail("The event table is empty")
            return

        # 1. Split -- depends only on timestamps and ground truth.
        split_result = split_dataset(events, labels, config)

        # 2. Fit the baseline on the training split alone.
        if baseline_dir is not None:
            model: BehavioralBaselineModel | None = BehavioralBaselineModel.load(
                baseline_dir, config.baseline
            )
        elif config.baseline.enabled:
            train_ids = split_result.event_ids_for(SplitLabel.TRAIN)
            training = [e for e in events if str(e.event_id) in train_ids]
            if not training:
                _fail("The training split is empty; cannot fit a baseline")
                return
            model = BehavioralBaselineModel(config.baseline).fit(
                training,
                permitted_event_ids=frozenset(e.event_id for e in training),
                interval=(
                    min(e.event_time for e in training),
                    max(e.event_time for e in training),
                ),
            )
        else:
            model = None

        # 3. Transform the entire stream, in temporal order.
        frame = FeatureEngine(config, built_catalog, baseline=model).run(events)

        # 4. Audit for leakage.
        audit = None
        if not skip_audit:
            audit = LeakageAuditor(config, built_catalog).audit(
                frame,
                events=events,
                labels=labels,
                assignments=list(split_result.assignments),
                baseline=model,
                baseline_artifact=None if model is None else model.artifact,
            )
        else:
            _err.print(
                "[yellow]Warning:[/yellow] the leakage audit was skipped; the "
                "timing contract has not been verified for this build"
            )

        ordered_labels = labels_for_events(
            [row[ANCHOR_EVENT_ID] for row in frame.rows], labels
        )
        feature_fingerprint = compute_features_fingerprint(frame)

        # 5. Stage, validate, and build the manifest.
        with tempfile.TemporaryDirectory(prefix="_pad_feat_build_") as staging_name:
            staging = Path(staging_name)
            write_feature_snapshots(frame, staging / "feature_snapshots.parquet")
            write_feature_labels(ordered_labels, staging / "feature_labels.parquet")
            write_feature_splits(
                split_result.assignments, staging / "feature_splits.parquet"
            )

            validation = FeatureValidator(built_catalog).validate_parquet(
                staging / "feature_snapshots.parquet",
                labels_path=staging / "feature_labels.parquet",
                splits_path=staging / "feature_splits.parquet",
            )

            manifest = build_feature_manifest(
                staging_dir=staging,
                catalog=built_catalog,
                config=config,
                feature_fingerprint=feature_fingerprint,
                source_dataset_fingerprint=compute_events_fingerprint(events),
                baseline_fingerprint=(
                    None if model is None else model.artifact.content_fingerprint
                ),
                row_count=len(frame.rows),
                label_count=len(ordered_labels),
                earliest_event_time=events[0].event_time,
                latest_event_time=events[-1].event_time,
                validation_status=str(validation.status),
                validation_result=validation.to_dict(),
                leakage_audit_result=None if audit is None else audit.to_dict(),
            )

        # 6. Publish, manifest last.
        published = FeaturePublisher(output_dir, overwrite=force).publish(
            frame, ordered_labels, split_result.assignments, manifest.to_dict()
        )

        # 7. Reports.
        report = generate_feature_quality_report(
            frame,
            feature_fingerprint=feature_fingerprint,
            assignments=list(split_result.assignments),
            class_distribution_by_split=split_result.manifest.class_distribution,
            exclusion_counts=split_result.manifest.exclusion_counts,
            validation_result=validation.to_dict(),
            leakage_audit_result=None if audit is None else audit.to_dict(),
        )
        if reports_dir is not None:
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "feature_quality.json").write_text(
                report_to_json(report) + "\n", encoding="utf-8"
            )
            (reports_dir / "feature_quality.md").write_text(
                report_to_markdown(report), encoding="utf-8"
            )
            if audit is not None:
                (reports_dir / "leakage_audit.json").write_text(
                    json_module.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                (reports_dir / "leakage_audit.md").write_text(
                    audit_result_to_markdown(audit), encoding="utf-8"
                )

        split_manifest_path = output_dir / "split_manifest.json"
        split_manifest_path.write_text(
            json_module.dumps(split_result.manifest.to_dict(), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    except _REPORTABLE as exc:
        _fail(str(exc))
        return
    except Exception as exc:
        _fail(f"Unexpected failure ({type(exc).__name__})")
        return

    _console.print(
        f"[green]Built[/green] {published.row_count:,} feature snapshots "
        f"x {published.feature_count} features"
    )
    _print_split_summary(split_result.manifest)
    _console.print(f"Feature fingerprint: {published.feature_fingerprint}")
    _console.print(f"Wrote artifacts to {_display(output_dir)}")

    if validation.status.value == "invalid":
        _err.print("[red]Feature validation failed.[/red]")
        raise typer.Exit(code=1) from None
    if audit is not None and not audit.passed:
        _err.print("[red]Leakage audit failed.[/red]")
        for message in audit.errors[:10]:
            _err.print(f"  {message}")
        raise typer.Exit(code=1) from None


# ---------------------------------------------------------------------------
# audit-leakage
# ---------------------------------------------------------------------------


@features_app.command(name="audit-leakage")
def audit_leakage_cmd(
    events_path: Annotated[
        Path, typer.Argument(help="Canonical authentication events Parquet file.")
    ],
    labels_path: Annotated[
        Path, typer.Option("--labels", help="Ground-truth labels Parquet file.")
    ],
    splits_path: Annotated[
        Path | None,
        typer.Option("--splits", help="Split assignments Parquet file."),
    ] = None,
    baseline_dir: Annotated[
        Path | None,
        typer.Option("--baseline", help="Directory holding the fitted baseline."),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Feature YAML configuration file.")
    ] = None,
    output_format: Annotated[
        str, typer.Option("--format", "-f", help="Output format: text or json.")
    ] = "text",
) -> None:
    """Run the twelve leakage checks against a dataset.

    Exposed separately from ``build`` on purpose: the checks are only useful
    as a gate if they can be run against an artifact someone hands you.
    """
    import json as json_module

    import pyarrow.parquet as pq

    from password_attack_detector.data.serialization import read_events_parquet
    from password_attack_detector.features.baselines import BehavioralBaselineModel
    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.engine import FeatureEngine
    from password_attack_detector.features.leakage import LeakageAuditor
    from password_attack_detector.features.serialization import (
        read_ground_truth_labels,
    )
    from password_attack_detector.features.splitting import (
        SplitAssignment,
        SplitLabel,
    )

    try:
        config = _load_config(config_path)
        built_catalog = build_catalog(config)
        events = read_events_parquet(events_path)
        labels = read_ground_truth_labels(labels_path)

        assignments = None
        if splits_path is not None:
            assignments = [
                SplitAssignment(
                    event_id=row["event_id"],
                    split=SplitLabel(row["split"]),
                    exclusion_reason=row["exclusion_reason"],
                )
                for row in pq.read_table(splits_path).to_pylist()
            ]

        model = (
            None
            if baseline_dir is None
            else BehavioralBaselineModel.load(baseline_dir, config.baseline)
        )
        frame = FeatureEngine(config, built_catalog, baseline=model).run(events)
        result = LeakageAuditor(config, built_catalog).audit(
            frame,
            events=events,
            labels=labels,
            assignments=assignments,
            baseline=model,
            baseline_artifact=None if model is None else model.artifact,
        )
    except _REPORTABLE as exc:
        _fail(str(exc))
        return
    except Exception as exc:
        _fail(f"Unexpected failure ({type(exc).__name__})")
        return

    if output_format == "json":
        _console.print(json_module.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        table = Table(title=f"Leakage audit: {result.status}")
        table.add_column("Check")
        table.add_column("Result")
        for check in result.checks:
            if check.skipped:
                verdict = "[yellow]skipped[/yellow]"
            elif check.passed:
                verdict = "[green]pass[/green]"
            else:
                verdict = "[red]FAIL[/red]"
            table.add_row(check.name, verdict)
        _console.print(table)
        for message in result.errors:
            _err.print(f"[red]{message}[/red]")

    if not result.passed:
        raise typer.Exit(code=1) from None


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@features_app.command()
def validate(
    snapshots_path: Annotated[
        Path, typer.Argument(help="Feature snapshots Parquet file.")
    ],
    labels_path: Annotated[
        Path | None,
        typer.Option("--labels", help="Feature labels Parquet file."),
    ] = None,
    splits_path: Annotated[
        Path | None,
        typer.Option("--splits", help="Feature splits Parquet file."),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Feature YAML configuration file.")
    ] = None,
) -> None:
    """Validate a feature table against its catalog."""
    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.validation import FeatureValidator

    try:
        config = _load_config(config_path)
        result = FeatureValidator(build_catalog(config)).validate_parquet(
            snapshots_path, labels_path=labels_path, splits_path=splits_path
        )
    except _REPORTABLE as exc:
        _fail(str(exc))
        return
    except Exception as exc:
        _fail(f"Unexpected failure ({type(exc).__name__})")
        return

    table = Table(title=f"Feature validation: {result.status}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Rows", f"{result.row_count:,}")
    table.add_row("Features", f"{result.feature_count:,}")
    table.add_row("Errors", str(len(result.errors)))
    table.add_row("Warnings", str(len(result.warnings)))
    _console.print(table)

    for finding in result.errors[:10]:
        _err.print(f"[red]{finding.code}[/red] {finding.message}")
    for finding in result.warnings[:5]:
        _console.print(f"[yellow]{finding.code}[/yellow] {finding.message}")

    if not result.passed:
        raise typer.Exit(code=1) from None


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@features_app.command()
def profile(
    events_path: Annotated[
        Path, typer.Argument(help="Canonical authentication events Parquet file.")
    ],
    labels_path: Annotated[
        Path | None, typer.Option("--labels", help="Ground-truth labels Parquet file.")
    ] = None,
    splits_path: Annotated[
        Path | None, typer.Option("--splits", help="Feature splits Parquet file.")
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Feature YAML configuration file.")
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Write JSON and Markdown reports."),
    ] = None,
) -> None:
    """Produce an aggregate quality report for a feature dataset."""
    import pyarrow.parquet as pq

    from password_attack_detector.data.serialization import read_events_parquet
    from password_attack_detector.features.catalog import build_catalog
    from password_attack_detector.features.engine import FeatureEngine
    from password_attack_detector.features.quality import (
        generate_feature_quality_report,
        report_to_json,
        report_to_markdown,
    )
    from password_attack_detector.features.serialization import (
        compute_features_fingerprint,
    )
    from password_attack_detector.features.splitting import (
        SplitAssignment,
        SplitLabel,
    )

    try:
        config = _load_config(config_path)
        events = read_events_parquet(events_path)
        frame = FeatureEngine(config, build_catalog(config)).run(events)

        assignments = None
        if splits_path is not None:
            assignments = [
                SplitAssignment(
                    event_id=row["event_id"],
                    split=SplitLabel(row["split"]),
                    exclusion_reason=row["exclusion_reason"],
                )
                for row in pq.read_table(splits_path).to_pylist()
            ]

        report = generate_feature_quality_report(
            frame,
            feature_fingerprint=compute_features_fingerprint(frame),
            assignments=assignments,
        )
    except _REPORTABLE as exc:
        _fail(str(exc))
        return
    except Exception as exc:
        _fail(f"Unexpected failure ({type(exc).__name__})")
        return

    if output_dir is None:
        _console.print(report_to_json(report))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_quality.json").write_text(
        report_to_json(report) + "\n", encoding="utf-8"
    )
    (output_dir / "feature_quality.md").write_text(
        report_to_markdown(report), encoding="utf-8"
    )
    _console.print(f"[green]Wrote reports to[/green] {_display(output_dir)}")


# ---------------------------------------------------------------------------
# verify-manifest
# ---------------------------------------------------------------------------


@features_app.command(name="verify-manifest")
def verify_manifest_cmd(
    directory: Annotated[
        Path, typer.Argument(help="Published feature dataset directory.")
    ],
) -> None:
    """Verify the integrity of a published feature dataset."""
    from password_attack_detector.features.manifest import verify_feature_dataset

    result = verify_feature_dataset(directory)

    table = Table(title="Feature manifest verification")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Detail")
    for check in result.checks:
        verdict = "[green]pass[/green]" if check.passed else "[red]FAIL[/red]"
        table.add_row(check.name, verdict, check.message)
    _console.print(table)

    if not result.passed:
        raise typer.Exit(code=1) from None
