#!/usr/bin/env python3
"""Verify a prepared serving state, and report what it holds.

Used twice, for two different reasons, and both of them matter.

**At build time**, immediately after ``prepare_demo_bundle.py`` runs and before
anything is pruned or copied into a runtime layer. A bundle that does not verify
must fail the *build*, because a Render deployment has no preparation step to
fall back on: whatever the image carries is what serves.

**At container start**, before the API is launched. The API verifies the bundle
again for itself -- that is its job and it is not delegated here -- but it does
so inside the process that would otherwise start serving. Checking first, in a
process that exits, means a corrupted or truncated image layer stops the
container with one legible message instead of surfacing as a permanently
un-ready service.

The check is :func:`~password_attack_detector.deployment.bundle
.load_serving_bundle`, unmodified. That function reads JSON payloads and
recomputes seals; it loads no fitted model and fits nothing, so running it costs
a few milliseconds and no meaningful memory.

**Nothing here can change what is served.** There is no option to name a
strategy, a model, a threshold, or a scope: the scope key is read from the
receipt the preparation wrote, and the strategy is read out of the bundle and
printed. A run that finds something other than ``stacked`` reports that fact and
still exits zero -- selecting a fusion strategy is the locked evaluation's
decision, and a verifier that rejected the answer would be making it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Where the prepared state lives inside every image this project builds.
DEFAULT_STATE_ROOT = "/srv/state"

#: Written last by ``prepare_demo_bundle.py``, and the only thing this script
#: takes direction from.
RECEIPT_FILE = "prepared.json"


class VerificationError(RuntimeError):
    """The prepared state is absent, incomplete, or does not verify."""


def _read_receipt(state: Path) -> dict[str, Any]:
    """Return the preparation receipt under *state*, or raise."""
    receipt = state / RECEIPT_FILE
    if not receipt.is_file():
        raise VerificationError(
            f"no preparation receipt at {receipt}; this state root was never "
            f"prepared, or was prepared by a different layout"
        )
    try:
        document: dict[str, Any] = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerificationError(
            f"the preparation receipt is unreadable ({type(exc).__name__})"
        ) from None
    for key in ("champion_scope_key", "artifact_root", "bundle_manifest"):
        if not document.get(key):
            raise VerificationError(f"the preparation receipt declares no {key!r}")
    return document


def verify(state: Path) -> dict[str, Any]:
    """Verify the bundle under *state* and return what it declares.

    Imported here rather than at module scope so ``--help`` and an absent-receipt
    failure cost nothing: the project package pulls in the scientific stack, and
    a script that cannot find its receipt should say so in milliseconds.
    """
    from password_attack_detector.deployment.bundle import load_serving_bundle

    receipt = _read_receipt(state)
    scope = str(receipt["champion_scope_key"])
    artifacts = state / str(receipt["artifact_root"])
    if not artifacts.is_dir():
        raise VerificationError("the receipt names an artifact root that is not there")

    manifest_path = state / str(receipt["bundle_manifest"])
    if not manifest_path.is_file():
        raise VerificationError(
            "the receipt names a serving bundle manifest that is not there"
        )

    try:
        bundle = load_serving_bundle(artifacts, scope_key=scope)
    except Exception as exc:
        raise VerificationError(
            f"the serving bundle does not verify ({type(exc).__name__}: {exc})"
        ) from None

    manifest = bundle.manifest
    return {
        "champion_scope_key": scope,
        "strategy": str(bundle.strategy),
        "stacked": bundle.stacked,
        "manifest_fingerprint": manifest.manifest_fingerprint,
        "stacked_state_fingerprint": manifest.stacked_state_fingerprint,
        "fusion_selection_fingerprint": manifest.fusion_selection_fingerprint,
        "champion_lock_fingerprint": manifest.champion_lock_fingerprint,
        "model_content_fingerprint": manifest.model_content_fingerprint,
        "binary_threshold_fingerprint": manifest.binary_threshold_fingerprint,
        "calibration_state_fingerprint": manifest.calibration_state_fingerprint,
        "feature_catalog_fingerprint": manifest.feature_catalog_fingerprint,
        "rule_configuration_fingerprint": manifest.rule_configuration_fingerprint,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the prepared state and return a process exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify a prepared serving bundle without loading a fitted model. "
            "Reports what the bundle declares; cannot change any of it."
        )
    )
    parser.add_argument(
        "--state-root",
        default=DEFAULT_STATE_ROOT,
        help="the prepared state root to verify (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the verified identity as JSON on stdout",
    )
    options = parser.parse_args(argv)

    try:
        identity = verify(Path(options.state_root).resolve())
    except VerificationError as failure:
        print(f"Serving bundle verification FAILED.\n{failure}", file=sys.stderr)
        return 1

    if options.json:
        print(json.dumps(identity, indent=2, sort_keys=True))
        return 0

    print("Serving bundle verified.")
    print(f"  champion scope         {identity['champion_scope_key']}")
    print(f"  fusion strategy        {identity['strategy']}")
    print(f"  manifest fingerprint   {identity['manifest_fingerprint']}")
    print(f"  stacked fingerprint    {identity['stacked_state_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
