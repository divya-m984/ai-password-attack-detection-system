#!/usr/bin/env python3
"""Build the demonstration's frozen scientific state, offline and once.

A serving process must never fit anything at startup, and a container image
cannot ship a trained champion: ``artifacts/`` and ``models/`` are untracked,
and a fitted model is not source. So the state has to be produced *before* the
API starts, by something that is explicitly not the API. This script is that
something.

What it runs is the project's real pipeline, one tracked CLI command at a time::

    data generate      synthetic events, from a seeded tracked configuration
    features build     point-in-time features, splits, and the manifest
    ml catalog         the reviewed feature allowlist, drafted from the catalog
    ml train           every enabled family, published as immutable runs
    ml select          a champion, chosen on validation evidence
    ml freeze-champion the champion, sealed
    ml predict         TRAIN, VALIDATION and TEST scored under the frozen model
    detection run      the Phase 4 rule engine over the same snapshots
    ml evaluate        the locked TEST evaluation, which selects the hybrid
    deploy materialize the serving bundle the API loads read-only

There is no shortcut in that list and no stand-in for any stage. Every artifact
the API serves is the output of the command that normally produces it, which is
the only way the container can claim to be running the real system.

**Nothing here decides anything this script could influence.** Which champion
wins and which fusion strategy is selected are outcomes of the commands above,
read from their published evidence. This script reports what they chose; it has
no option to name a model, move a threshold, or ask for a strategy, and if a run
selects something other than ``stacked`` that is a fact to report rather than a
failure to retry differently.

Re-running is a no-op once a bundle is published: the receipt is checked first
and the pipeline is skipped. ``--force`` starts over from an empty state root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

#: Where this script writes, unless told otherwise. Inside the container this is
#: the mount point of the named volume the API later reads read-only.
DEFAULT_STATE_ROOT: Final[str] = "/srv/state"

#: The tracked configurations the demonstration deployment is built from. Each
#: is a reviewed file in the repository, and none of them is generated here.
SYNTHETIC_CONFIG: Final[str] = "configs/data/synthetic-demo.yaml"
FEATURE_CONFIG: Final[str] = "configs/features/feature-demo.yaml"
RULE_CONFIG: Final[str] = "configs/detection/rules-demo.yaml"
ML_CONFIG: Final[str] = "configs/ml/model-demo.yaml"

#: Written last, and only on success. Its presence is what makes a second run a
#: no-op, so a partially built state root is never mistaken for a prepared one.
RECEIPT_FILE: Final[str] = "prepared.json"

#: The receipt's own version, so a future change to this layout is detectable
#: rather than silently reused by an API that expects the old one.
RECEIPT_SCHEMA_VERSION: Final[str] = "1.0.0"


class PreparationError(RuntimeError):
    """A pipeline stage failed, or produced something the next stage cannot use."""


def _configuration_root() -> Path:
    """Return the directory the tracked configurations are read from.

    Located by the presence of the configurations themselves rather than by
    ``pyproject.toml``: in a checkout this script sits beside the sources, and in
    the runtime image it sits beside ``configs/`` with no source tree at all.
    Looking for what is actually needed makes both layouts the same question.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent, *here.parents):
        if (candidate / SYNTHETIC_CONFIG).is_file():
            return candidate
    raise PreparationError(
        f"cannot locate the tracked configurations: no ancestor of this script "
        f"holds {SYNTHETIC_CONFIG}"
    )


def _run(step: str, arguments: Sequence[str], *, root: Path) -> str:
    """Run one CLI command, or raise :class:`PreparationError` with its output.

    The command is invoked through ``python -m password_attack_detector`` so the
    pipeline runs the installed package's real entry point rather than anything
    this file assembles.
    """
    started = time.monotonic()
    command = [sys.executable, "-m", "password_attack_detector", *arguments]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"},
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise PreparationError(
            f"{step} failed with exit code {completed.returncode}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    print(f"  {step:<22} ok  ({elapsed:5.1f}s)", flush=True)
    return completed.stdout


def _flatten(arguments: dict[str, str]) -> list[str]:
    """Return ``{option: value}`` as a flat argument list."""
    flat: list[str] = []
    for option, value in arguments.items():
        flat += [option, value]
    return flat


def _prediction_ids(artifacts: Path) -> dict[str, str]:
    """Return every published prediction identifier, keyed by the split it scored."""
    from password_attack_detector.ml.prediction_manifest import (
        PREDICTION_MANIFEST_FILE,
        PREDICTIONS_DIR,
    )

    found: dict[str, str] = {}
    directory = artifacts / PREDICTIONS_DIR
    if not directory.is_dir():
        raise PreparationError("no predictions were published")
    for item in sorted(directory.iterdir()):
        manifest = item / PREDICTION_MANIFEST_FILE
        if manifest.is_file():
            scope = json.loads(manifest.read_text(encoding="utf-8"))["scope"]
            found[str(scope)] = item.name
    missing = {"validation", "test"} - set(found)
    if missing:
        raise PreparationError(f"no prediction was published for {sorted(missing)}")
    return found


def _frozen_selection(artifacts: Path) -> dict[str, Any]:
    """Return the frozen fusion selection the locked TEST evaluation published."""
    reports = sorted((artifacts / "evaluations").glob("*/system_comparison.json"))
    if not reports:
        raise PreparationError("the locked evaluation published no system comparison")
    payload: dict[str, Any] = json.loads(reports[-1].read_text(encoding="utf-8"))
    selection: dict[str, Any] = payload["fusion"]
    return selection


def _champion_scope(artifacts: Path) -> str:
    """Return the frozen champion's scope key."""
    locks = sorted((artifacts / "champion").glob("*/champion.lock"))
    if not locks:
        raise PreparationError("no champion was frozen")
    return locks[-1].parent.name


def _receipt_path(state: Path) -> Path:
    """Return where the completion receipt lives."""
    return state / RECEIPT_FILE


def already_prepared(state: Path) -> dict[str, Any] | None:
    """Return the receipt of a completed preparation under *state*, or ``None``.

    A receipt whose bundle manifest is missing is treated as absent: the state
    root may have been rebuilt underneath it, and re-running is cheap where
    serving a half-published bundle is not.
    """
    receipt = _receipt_path(state)
    if not receipt.is_file():
        return None
    try:
        document: dict[str, Any] = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if document.get("receipt_schema_version") != RECEIPT_SCHEMA_VERSION:
        return None
    manifest = state / str(document.get("bundle_manifest", ""))
    if not manifest.is_file():
        return None
    return document


def prepare(state: Path, *, root: Path) -> dict[str, Any]:
    """Run the whole pipeline into *state* and return the completion receipt."""
    dataset = state / "dataset"
    processed = state / "processed"
    artifacts = state / "artifacts"
    detection = state / "detection"
    reports = state / "reports"
    allowlist = state / "allowlist.yaml"
    for directory in (state, reports):
        directory.mkdir(parents=True, exist_ok=True)

    synthetic = root / SYNTHETIC_CONFIG
    features = root / FEATURE_CONFIG
    rules = root / RULE_CONFIG
    ml = root / ML_CONFIG
    for configuration in (synthetic, features, rules, ml):
        if not configuration.is_file():
            raise PreparationError(f"missing tracked configuration: {configuration}")

    print("Preparing the demonstration deployment. Nothing here is served yet.")
    print(f"  state root             {state}", flush=True)

    _run(
        "data generate",
        ["data", "generate", str(synthetic), "-o", str(dataset)],
        root=root,
    )
    _run(
        "features build",
        [
            "features",
            "build",
            str(dataset / "events.parquet"),
            "--labels",
            str(dataset / "labels.parquet"),
            "--config",
            str(features),
            "-o",
            str(processed),
            "--reports-dir",
            str(reports),
        ],
        root=root,
    )
    _run(
        "ml catalog",
        [
            "ml",
            "catalog",
            "--emit-allowlist",
            str(allowlist),
            "--feature-config",
            str(features),
        ],
        root=root,
    )

    shared = {
        "--features": str(processed / "feature_snapshots.parquet"),
        "--splits": str(processed / "feature_splits.parquet"),
        "--allowlist": str(allowlist),
        "--feature-config": str(features),
        "--config": str(ml),
        "--output-root": str(artifacts),
    }

    _run(
        "ml train",
        [
            "ml",
            "train",
            *_flatten(
                {
                    **shared,
                    "--labels": str(processed / "feature_labels.parquet"),
                    "--campaign-labels": str(dataset / "labels.parquet"),
                    "--feature-manifest": str(processed / "feature_manifest.json"),
                }
            ),
        ],
        root=root,
    )
    _run(
        "ml select",
        [
            "ml",
            "select",
            "--output-root",
            str(artifacts),
            "--config",
            str(ml),
            "--reports-dir",
            str(reports),
        ],
        root=root,
    )
    _run(
        "ml freeze-champion",
        ["ml", "freeze-champion", "--output-root", str(artifacts), "--config", str(ml)],
        root=root,
    )

    for split in ("train", "validation", "test"):
        _run(
            f"ml predict [{split}]",
            [
                "ml",
                "predict",
                *_flatten(
                    {
                        **shared,
                        "--feature-manifest": str(processed / "feature_manifest.json"),
                        "--split": split,
                    }
                ),
            ],
            root=root,
        )

    _run(
        "detection run",
        [
            "detection",
            "run",
            "--config",
            str(rules),
            "--features",
            str(processed / "feature_snapshots.parquet"),
            "--feature-manifest",
            str(processed / "feature_manifest.json"),
            "--feature-config",
            str(features),
            "-o",
            str(detection),
            "--reports-dir",
            str(detection / "reports"),
        ],
        root=root,
    )

    from password_attack_detector.detection.serialization import RISK_FILE

    identifiers = _prediction_ids(artifacts)
    _run(
        "ml evaluate",
        [
            "ml",
            "evaluate",
            *_flatten(
                {
                    **shared,
                    "--labels": str(processed / "feature_labels.parquet"),
                    "--campaign-labels": str(dataset / "labels.parquet"),
                    "--risk-assessments": str(detection / RISK_FILE),
                    "--reports-dir": str(reports),
                    "--prediction": identifiers["test"],
                    "--validation-prediction": identifiers["validation"],
                }
            ),
        ],
        root=root,
    )

    # No ``--rule-config`` here, and that is not an omission. ``ml evaluate``
    # passes none either, so the frozen selection's rule-configuration
    # fingerprint is a default DetectionConfig's. Supplying one here would be a
    # genuinely different upstream input and the materializer would refuse it as
    # ``rule_configuration_mismatch`` -- the check working, not a bug.
    _run(
        "deploy materialize",
        [
            "deploy",
            "materialize",
            *_flatten(
                {
                    **shared,
                    "--labels": str(processed / "feature_labels.parquet"),
                    "--campaign-labels": str(dataset / "labels.parquet"),
                    "--risk-assessments": str(detection / RISK_FILE),
                    "--validation-prediction": identifiers["validation"],
                }
            ),
        ],
        root=root,
    )

    from password_attack_detector.deployment.bundle import (
        BUNDLE_MANIFEST_FILE,
        SERVING_BUNDLE_DIR,
    )

    scope = _champion_scope(artifacts)
    manifest = artifacts / SERVING_BUNDLE_DIR / scope / BUNDLE_MANIFEST_FILE
    if not manifest.is_file():
        raise PreparationError("the serving bundle was not published")
    selection = _frozen_selection(artifacts)

    receipt = {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "champion_scope_key": scope,
        "selected_fusion_strategy": selection.get("selected_strategy"),
        "fusion_selection_status": selection.get("status"),
        # Relative to the state root on purpose: a receipt that recorded an
        # absolute path would publish the layout of whatever machine built it.
        "bundle_manifest": str(manifest.relative_to(state)),
        "artifact_root": str(artifacts.relative_to(state)),
        "allowlist": str(allowlist.relative_to(state)),
        "synthetic_config": SYNTHETIC_CONFIG,
        "feature_config": FEATURE_CONFIG,
        "rule_config": RULE_CONFIG,
        "ml_config": ML_CONFIG,
    }
    _receipt_path(state).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _report(receipt: dict[str, Any]) -> None:
    """Print what the pipeline decided, without printing where anything lives."""
    print("")
    print("Prepared. What the pipeline decided, not what this script chose:")
    print(f"  champion scope         {receipt['champion_scope_key']}")
    print(f"  fusion selection       {receipt['fusion_selection_status']}")
    print(f"  selected strategy      {receipt['selected_fusion_strategy']}")
    print("")


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare the serving state, and return a process exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the demonstration's frozen scientific state offline, "
            "before any service starts. Runs the project's real pipeline."
        )
    )
    parser.add_argument(
        "--state-root",
        default=os.environ.get("PAD_DEMO_STATE_ROOT", DEFAULT_STATE_ROOT),
        help="directory the pipeline writes into (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard an existing prepared state and build it again",
    )
    options = parser.parse_args(argv)

    state = Path(options.state_root).resolve()
    root = _configuration_root()

    if options.force and state.exists():
        print("Discarding the existing state root, as --force was given.")
        shutil.rmtree(state)

    existing = already_prepared(state)
    if existing is not None:
        print("The serving state is already prepared; nothing to do.")
        _report(existing)
        return 0

    started = time.monotonic()
    try:
        receipt = prepare(state, root=root)
    except PreparationError as failure:
        print(f"\nPreparation failed.\n{failure}", file=sys.stderr)
        return 1
    print(f"\nPipeline finished in {time.monotonic() - started:.1f}s.")
    _report(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
