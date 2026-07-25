"""Typer CLI for the Password Attack Detector.

Available commands:

    password-attack-detector version
    password-attack-detector doctor
    password-attack-detector show-config
"""

from __future__ import annotations

import sys
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from password_attack_detector import __version__

app = typer.Typer(
    name="password-attack-detector",
    help="AI-powered defensive system for detecting password-based attacks.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


@app.command()
def version() -> None:
    """Print the current package version."""
    typer.echo(f"password-attack-detector {__version__}")


@app.command()
def doctor() -> None:
    """Run system health checks and report pass/fail status for each."""
    checks: list[tuple[str, bool, str]] = []

    # 1. Python version
    py_ok = sys.version_info >= (3, 12)
    checks.append(("Python >= 3.12", py_ok, sys.version.split()[0]))

    # 2. Package importability
    try:
        import password_attack_detector  # noqa: F401

        checks.append(("Package importable", True, "password_attack_detector"))
    except ImportError as exc:
        checks.append(("Package importable", False, str(exc)))

    # 3. Configuration loading
    settings: Any = None
    try:
        from password_attack_detector.config import load_settings

        settings = load_settings()
        checks.append(("Config loads", True, f"env={settings.environment}"))
    except Exception as exc:
        checks.append(("Config loads", False, str(exc)))

    # 4. Data directory accessible
    if settings is not None:
        data_dir = settings.data_dir
        accessible = data_dir is not None and data_dir.exists()
        checks.append(("Data dir exists", accessible, str(data_dir)))
    else:
        checks.append(("Data dir exists", False, "config did not load"))

    # 5. Artifacts directory accessible
    if settings is not None:
        artifacts_dir = settings.artifacts_dir
        accessible = artifacts_dir is not None and artifacts_dir.exists()
        checks.append(("Artifacts dir exists", accessible, str(artifacts_dir)))
    else:
        checks.append(("Artifacts dir exists", False, "config did not load"))

    table = Table(
        title="System Health Check",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Detail", style="dim")

    all_passed = True
    for name, passed, detail in checks:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, status, detail)
        if not passed:
            all_passed = False

    console.print(table)

    if not all_passed:
        raise typer.Exit(code=1)


@app.command(name="show-config")
def show_config() -> None:
    """Display the current configuration. Secret values are redacted."""
    from pydantic import SecretStr

    from password_attack_detector.config import load_settings

    settings = load_settings()

    lines: list[str] = []
    for field_name, field_info in type(settings).model_fields.items():
        # Skip excluded fields entirely (e.g. pseudonymization_key).
        # Such fields must not appear in output even as a masked placeholder.
        if getattr(field_info, "exclude", False):
            continue
        value = getattr(settings, field_name)
        display = "**REDACTED**" if isinstance(value, SecretStr) else str(value)
        lines.append(f"[cyan]{field_name}[/cyan] = {display}")

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold]Current Configuration[/bold]",
            border_style="blue",
        )
    )
