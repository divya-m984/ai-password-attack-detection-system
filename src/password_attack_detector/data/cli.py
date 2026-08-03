"""Data engineering CLI command group.

Subcommands:

    password-attack-detector data generate        — generate synthetic dataset
    password-attack-detector data ingest          — ingest CSV or JSONL
    password-attack-detector data validate        — validate canonical Parquet
    password-attack-detector data profile         — generate quality reports
    password-attack-detector data manifest        — create manifest for dataset
    password-attack-detector data verify-manifest — verify manifest integrity
    password-attack-detector data schema          — print canonical schema
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from password_attack_detector.exceptions import (
    DataValidationError,
    IngestionError,
    PseudonymizationError,
)

data_app = typer.Typer(
    name="data",
    help="Data engineering: generate, ingest, validate, profile, and publish datasets.",
    no_args_is_help=True,
)

_console = Console()
_err = Console(stderr=True)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@data_app.command()
def generate(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to synthetic YAML configuration file."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o", help="Output directory for the generated dataset."
        ),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing output artifacts."),
    ] = False,
) -> None:
    """Generate a synthetic authentication-event dataset from a YAML config.

    Writes canonical events.parquet, labels.parquet, events.jsonl,
    quality-report.json, quality-report.md, and manifest.json using
    staged publication with automatic rollback on failure.

    Synthetic data never uses the pseudonymization key.
    """
    from password_attack_detector.data.serialization import DatasetPublisher
    from password_attack_detector.data.synthetic.config import SyntheticConfig
    from password_attack_detector.data.synthetic.generator import generate_dataset

    if not config_path.exists():
        _err.print(f"[red]Config file not found:[/red] {config_path.name}")
        raise typer.Exit(code=1) from None

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("YAML root must be a mapping")
        config = SyntheticConfig.model_validate(raw)
    except Exception as exc:
        _err.print(f"[red]Invalid configuration:[/red] {exc}")
        raise typer.Exit(code=1) from None

    try:
        result = generate_dataset(config)
        publisher = DatasetPublisher(output_dir.resolve(), overwrite=force)
        published = publisher.publish(result)
    except DataValidationError as exc:
        _err.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        _err.print(f"[red]Unexpected error:[/red] {type(exc).__name__}")
        raise typer.Exit(code=1) from None

    _console.print("[green]Dataset generated successfully.[/green]")
    _console.print(f"  Events:           {published.num_events:,}")
    _console.print(f"  Labels:           {published.num_labels:,}")
    _console.print(f"  Content hash:     {published.content_fingerprint[:16]}…")
    _console.print(f"  Output directory: {_display(published.events_path.parent)}/")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@data_app.command()
def ingest(
    source_path: Annotated[
        Path,
        typer.Argument(help="Path to the source CSV or JSONL file."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o", help="Output directory for published events."
        ),
    ],
    field_map: Annotated[
        list[str],
        typer.Option(
            "--field-map",
            "-m",
            help="Source-to-canonical field mapping: SOURCE=CANONICAL (repeatable).",
        ),
    ] = [],  # noqa: B006
    fmt: Annotated[
        str | None,
        typer.Option(
            "--format",
            "-f",
            help="Input format: csv or jsonl. Auto-detected from file extension.",
        ),
    ] = None,
    policy: Annotated[
        str,
        typer.Option("--policy", "-p", help="Invalid-row policy: fail or quarantine."),
    ] = "fail",
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing output artifacts."),
    ] = False,
) -> None:
    """Ingest a CSV or JSONL file of authentication events into canonical Parquet.

    Requires the PAD_PSEUDONYMIZATION_KEY environment variable. Source
    identifiers (user_id, source_id, device_id, session_id) are pseudonymized
    via HMAC-SHA256 before storage. Prohibited sensitive fields are rejected.
    """
    from password_attack_detector.config import load_settings
    from password_attack_detector.data.ingestion import (
        CSVIngestionAdapter,
        InvalidRowPolicy,
        JSONLIngestionAdapter,
    )
    from password_attack_detector.data.manifest import build_directory_manifest
    from password_attack_detector.data.privacy import PseudonymService
    from password_attack_detector.data.serialization import (
        compute_events_fingerprint,
        write_events_jsonl,
        write_events_parquet,
    )
    from password_attack_detector.data.validation import DatasetValidator

    # Detect format from file extension or --format flag.
    suffix = source_path.suffix.lower()
    detected = fmt or (
        "csv"
        if suffix == ".csv"
        else "jsonl"
        if suffix in (".jsonl", ".ndjson")
        else None
    )
    if detected not in ("csv", "jsonl"):
        _err.print(
            "[red]Cannot detect format.[/red] Specify --format csv or --format jsonl."
        )
        raise typer.Exit(code=1) from None

    if not source_path.exists():
        _err.print(f"[red]Input file not found:[/red] {source_path.name}")
        raise typer.Exit(code=1) from None

    try:
        row_policy = InvalidRowPolicy(policy.lower())
    except ValueError:
        _err.print(
            f"[red]Unknown policy:[/red] {policy!r}. Use 'fail' or 'quarantine'."
        )
        raise typer.Exit(code=1) from None

    # Parse SOURCE=CANONICAL field-map entries.
    parsed_map: dict[str, str] = {}
    for entry in field_map:
        if "=" not in entry:
            _err.print(
                f"[red]Invalid --field-map entry:[/red] {entry!r}. "
                "Use SOURCE=CANONICAL."
            )
            raise typer.Exit(code=1) from None
        src, _, canonical = entry.partition("=")
        parsed_map[src.strip()] = canonical.strip()

    # Load settings and build PseudonymService (requires PAD_PSEUDONYMIZATION_KEY).
    try:
        settings = load_settings()
        svc = PseudonymService.from_settings(settings)
    except PseudonymizationError as exc:
        _err.print(f"[red]Missing pseudonymization key:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        _err.print(f"[red]Configuration error:[/red] {type(exc).__name__}")
        raise typer.Exit(code=1) from None

    # Run ingestion adapter.
    try:
        if detected == "csv":
            adapter: CSVIngestionAdapter | JSONLIngestionAdapter = CSVIngestionAdapter(
                policy=row_policy,
                pseudonym_service=svc,
                extra_field_map=parsed_map or None,
            )
        else:
            adapter = JSONLIngestionAdapter(
                policy=row_policy,
                pseudonym_service=svc,
                extra_field_map=parsed_map or None,
            )
        ingest_result = adapter.ingest(source_path)
    except IngestionError as exc:
        _err.print(f"[red]Ingestion rejected:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        _err.print(f"[red]Unexpected ingestion error:[/red] {type(exc).__name__}")
        raise typer.Exit(code=1) from None

    if ingest_result.accepted_count == 0:
        diag = ingest_result.empty_diagnostic
        reason = diag.message if diag else "No events accepted"
        _err.print(f"[red]Empty ingestion — nothing published:[/red] {reason}")
        raise typer.Exit(code=1) from None

    out = output_dir.resolve()
    events_path = out / "events.parquet"
    jsonl_path = out / "events.jsonl"
    manifest_path = out / "manifest.json"

    existing = [p for p in (events_path, jsonl_path, manifest_path) if p.exists()]
    if existing and not force:
        _err.print("[red]Output already exists.[/red] Use --force to overwrite.")
        raise typer.Exit(code=1) from None

    out.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=out.parent, prefix="_pad_ingest_stage_"))
    content_fp: str = ""
    try:
        events = ingest_result.events
        write_events_parquet(events, staging / "events.parquet")
        write_events_jsonl(events, staging / "events.jsonl")

        content_fp = compute_events_fingerprint(events)
        val_result = DatasetValidator().validate_parquet(staging / "events.parquet")
        mfst = build_directory_manifest(
            staging,
            events=events,
            source_type="ingested",
            labels_count=0,
            validation_status=str(val_result.status),
            content_fingerprint=content_fp,
        )
        (staging / "manifest.json").write_text(
            json.dumps(mfst.to_dict(), indent=2), encoding="utf-8"
        )

        for fname in ("events.parquet", "events.jsonl", "manifest.json"):
            shutil.move(str(staging / fname), str(out / fname))

    except Exception as exc:
        _err.print(f"[red]Publication failed:[/red] {type(exc).__name__}")
        raise typer.Exit(code=1) from None
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    _console.print("[green]Ingestion complete.[/green]")
    _console.print(f"  Accepted:         {ingest_result.accepted_count:,}")
    _console.print(f"  Quarantined:      {ingest_result.quarantine_count:,}")
    _console.print(f"  Content hash:     {content_fp[:16]}…")
    _console.print(f"  Output directory: {_display(out)}/")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@data_app.command()
def validate(
    events_path: Annotated[
        Path,
        typer.Argument(help="Path to canonical events.parquet file."),
    ],
) -> None:
    """Validate a canonical events Parquet file against the AuthEvent schema.

    Exits 0 when validation passes or produces only warnings.
    Exits 1 on schema errors or when the file cannot be read.
    """
    from password_attack_detector.data.validation import (
        DatasetValidator,
        ValidationStatus,
    )

    if not events_path.exists():
        _err.print(f"[red]File not found:[/red] {events_path.name}")
        raise typer.Exit(code=1) from None

    try:
        result = DatasetValidator().validate_parquet(events_path)
    except Exception as exc:
        _err.print(f"[red]Validation error:[/red] {type(exc).__name__}")
        raise typer.Exit(code=1) from None

    if result.status == ValidationStatus.VALID:
        _console.print(
            f"[green]PASS[/green]  status={result.status}  rows={result.record_count:,}"
        )
    elif result.status == ValidationStatus.WARNING:
        _console.print(
            f"[yellow]WARN[/yellow]  status={result.status}  rows={result.record_count:,}"
        )
        for err in result.errors[:5]:
            _console.print(f"  [{err.code}] {err.field}: {err.message}")
    else:
        _err.print(
            f"[red]FAIL[/red]  status={result.status}  rows={result.record_count:,}"
            f"  errors={result.rejected_count}"
        )
        for err in result.errors[:10]:
            _err.print(f"  [{err.code}] {err.field}: {err.message}")
        raise typer.Exit(code=1) from None


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@data_app.command()
def profile(
    events_path: Annotated[
        Path,
        typer.Argument(help="Path to canonical events.parquet file."),
    ],
    gt_path: Annotated[
        Path | None,
        typer.Option("--gt-path", help="Optional path to ground-truth labels.parquet."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Write quality-report.json and quality-report.md here. "
            "Prints JSON to stdout when omitted.",
        ),
    ] = None,
) -> None:
    """Generate JSON and Markdown quality reports for a canonical dataset.

    Reports never contain raw event identifiers, pseudonym values, or secrets.
    """
    from password_attack_detector.data.quality import (
        generate_quality_report,
        report_to_json,
        report_to_markdown,
    )

    if not events_path.exists():
        _err.print(f"[red]Events file not found:[/red] {events_path.name}")
        raise typer.Exit(code=1) from None

    if gt_path is not None and not gt_path.exists():
        _err.print(f"[red]Ground-truth file not found:[/red] {gt_path.name}")
        raise typer.Exit(code=1) from None

    try:
        report = generate_quality_report(events_path, gt_path=gt_path)
    except DataValidationError as exc:
        _err.print(f"[red]Profile error:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        _err.print(f"[red]Unexpected error:[/red] {type(exc).__name__}")
        raise typer.Exit(code=1) from None

    json_str = report_to_json(report)
    md_str = report_to_markdown(report)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "quality-report.json").write_text(json_str, encoding="utf-8")
        (output_dir / "quality-report.md").write_text(md_str, encoding="utf-8")
        _console.print(
            f"[green]Profile complete.[/green] "
            f"Reports written to {_display(output_dir)}/"
        )
    else:
        _console.print(json_str)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


@data_app.command()
def manifest(
    directory: Annotated[
        Path,
        typer.Argument(help="Dataset directory containing events.parquet."),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing manifest.json."),
    ] = False,
) -> None:
    """Create manifest.json for an existing validated dataset directory.

    Computes SHA-256 checksums for all artifact files found in the directory,
    validates the events Parquet, and writes a complete manifest with
    fingerprint and reproducibility metadata.
    """
    import pyarrow.parquet as pq

    from password_attack_detector.data.manifest import build_directory_manifest
    from password_attack_detector.data.serialization import (
        compute_events_fingerprint,
        read_events_parquet,
    )
    from password_attack_detector.data.validation import DatasetValidator

    events_file = directory / "events.parquet"
    manifest_file = directory / "manifest.json"

    if not directory.exists():
        _err.print("[red]Directory not found.[/red]")
        raise typer.Exit(code=1) from None

    if not events_file.exists():
        _err.print("[red]events.parquet not found in directory.[/red]")
        raise typer.Exit(code=1) from None

    if manifest_file.exists() and not force:
        _err.print("[red]manifest.json already exists.[/red] Use --force to overwrite.")
        raise typer.Exit(code=1) from None

    try:
        events = read_events_parquet(events_file)
    except DataValidationError as exc:
        _err.print(f"[red]Cannot read events:[/red] {exc}")
        raise typer.Exit(code=1) from None

    try:
        val_result = DatasetValidator().validate_parquet(events_file)
        content_fp = compute_events_fingerprint(events)

        labels_count = 0
        labels_file = directory / "labels.parquet"
        if labels_file.exists():
            labels_count = pq.read_table(labels_file).num_rows

        mfst = build_directory_manifest(
            directory,
            events=events,
            source_type="ingested",
            labels_count=labels_count,
            validation_status=str(val_result.status),
            content_fingerprint=content_fp,
        )
        manifest_file.write_text(json.dumps(mfst.to_dict(), indent=2), encoding="utf-8")
    except Exception as exc:
        _err.print(f"[red]Manifest creation failed:[/red] {type(exc).__name__}")
        raise typer.Exit(code=1) from None

    _console.print(
        f"[green]Manifest created.[/green] {_display(directory)}/manifest.json"
    )
    _console.print(f"  Row count:    {len(events):,}")
    _console.print(f"  Content hash: {content_fp[:16]}…")


# ---------------------------------------------------------------------------
# verify-manifest
# ---------------------------------------------------------------------------


@data_app.command(name="verify-manifest")
def verify_manifest_cmd(
    directory: Annotated[
        Path,
        typer.Argument(help="Dataset directory containing manifest.json."),
    ],
) -> None:
    """Verify the integrity of a published dataset manifest.

    Runs all 10 verification checks (checksum, schema version, row count,
    path safety, temporal consistency, etc.) and exits 0 only when all pass.
    """
    from password_attack_detector.data.manifest import verify_dataset

    if not directory.exists():
        _err.print("[red]Directory not found.[/red]")
        raise typer.Exit(code=1) from None

    result = verify_dataset(directory)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Detail", style="dim")

    for check in result.checks:
        status = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
        table.add_row(check.name, status, check.message)

    _console.print(table)

    if result.passed:
        _console.print("[green]Verification passed.[/green]")
    else:
        _err.print("[red]Verification failed.[/red]")
        raise typer.Exit(code=1) from None


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


@data_app.command()
def schema(
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: text or json."),
    ] = "text",
) -> None:
    """Print the canonical AuthEvent schema with field names and types.

    Does not expose configuration secrets, pseudonymization keys, or event data.
    """
    from password_attack_detector.data.schemas import SCHEMA_VERSION, AuthEvent

    if fmt not in ("text", "json"):
        _err.print(f"[red]Unknown format:[/red] {fmt!r}. Use 'text' or 'json'.")
        raise typer.Exit(code=1) from None

    fields = AuthEvent.model_fields

    if fmt == "json":
        schema_dict: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "model": "AuthEvent",
            "fields": {
                name: {
                    "type": str(info.annotation),
                    "required": info.is_required(),
                    "default": None if info.is_required() else repr(info.default),
                }
                for name, info in fields.items()
            },
        }
        _console.print(json.dumps(schema_dict, indent=2))
    else:
        _console.print(
            f"[bold]AuthEvent schema[/bold]  (schema_version={SCHEMA_VERSION})"
        )
        _console.print()
        table = Table(show_header=True, header_style="bold")
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Type")
        table.add_column("Required", justify="center")
        table.add_column("Default")
        for name, info in fields.items():
            req = "yes" if info.is_required() else "no"
            dflt = "" if info.is_required() else repr(info.default)
            table.add_row(name, str(info.annotation), req, dflt)
        _console.print(table)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _display(path: Path) -> str:
    """Return path relative to cwd when possible, else just the final component."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name
