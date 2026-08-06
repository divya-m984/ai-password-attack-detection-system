"""Machine-learning CLI command group.

Subcommands::

    password-attack-detector ml catalog  -- print the versioned model catalog

Milestone 1 ships exactly one command.  Training, calibration, threshold
selection, inference, evaluation, and comparison arrive in later milestones,
and no placeholder is registered for them: a command that exists but does
nothing is worse than one that is honestly absent, because ``--help`` would
advertise a capability the code does not have.

**No command prints an identifier.**  Not an event identifier, an entity
pseudonym, a coordinate, a secret, or an absolute path.  Output is metadata:
model identifiers, versions, families, declared hyperparameters, eligibility,
and limitations.  No executable configuration and no source code is emitted,
and no measured performance figure appears anywhere -- this command describes
what *may* be fitted, never what was.

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
    ManifestVerificationError,
    MLConfigurationError,
    ModelNotReadyError,
    ModelSerializationError,
    ModelTrainingError,
)

ml_app = typer.Typer(
    name="ml",
    help=(
        "Machine-learning detection layer: inspect the model catalog. "
        "Models are fitted on Phase 3 feature snapshots and are reported "
        "alongside the rule engine, never in place of it."
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
) -> None:
    """Print the versioned model catalog.

    Rendered from the same ``MODEL_CATALOG`` the training path will read, so
    the documentation cannot describe a model family the code does not declare.

    Catalog membership is not championship: a family listed here may be fitted
    and evaluated, but promotion additionally requires proven serializer and
    inference-adapter parity plus every validation gate. Nothing printed here
    is a measured result.
    """
    from password_attack_detector.ml.catalog import (
        MODEL_CATALOG,
        MODEL_CATALOG_VERSION,
        model_catalog_to_markdown,
    )

    if output_format not in {"text", "markdown"}:
        _fail(f"Unknown format {output_format!r}; use 'text' or 'markdown'")

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
