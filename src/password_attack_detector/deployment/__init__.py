"""Deployment artifacts: making a frozen scientific decision *servable*.

Phase 5 decides things and seals the decisions.  Phase 6 serves them.  Between
those two there is a gap that only shows up for one strategy: ``STACKED``.

``OR_GATE`` and ``AND_GATE`` are pure functions of two booleans, so a serving
process that knows *which* strategy was selected can execute it immediately.
``STACKED`` is a **fitted** meta-learner, and Phase 5 records only its
fingerprint -- the state itself lived in the memory of the ``ml evaluate`` run
that fitted it and was never published as a standalone artifact.  A deployment
whose selection was stacked could therefore name its hybrid and not run it.

This package closes that gap without moving anything Phase 5 froze:

* :mod:`~password_attack_detector.deployment.materialize` **reconstructs** the
  state offline, from the same pre-TEST lineage the original fit consumed, and
  refuses to publish unless the reconstruction recomputes the fingerprint Phase 5
  already sealed.
* :mod:`~password_attack_detector.deployment.bundle` is the artifact contract the
  verified state is published into, alongside the frozen fusion selection and the
  complete model, preprocessor, calibrator, threshold, and fusion lineage.

Three properties hold by construction, and are tested rather than asserted:

**Nothing here fits at serving time.**  The reconstruction is an offline
operator action with its own command.  The serving runtime only ever *loads* a
published bundle and verifies it; the API's own import-time guard refuses to let
a fitting function into the serving module's namespace.

**Nothing here reads a TEST label.**  Reconstruction consumes TRAIN rows
(out-of-fold) and validation-B rows, exactly as the original fit did.  The TEST
ground-truth reader is gated behind a proof object this package never
constructs, and mutating the TEST labels leaves the materialized state
byte-identical.

**Nothing here re-decides anything.**  No reselection, no new threshold, no new
fusion threshold, no fallback strategy.  A reconstruction that disagrees with the
frozen fingerprint is a refusal, not a new champion.
"""

from __future__ import annotations

from password_attack_detector.deployment.bundle import (
    BUNDLE_MANIFEST_FILE,
    BUNDLE_SCHEMA_VERSION,
    FUSION_SELECTION_FILE,
    SERVING_BUNDLE_DIR,
    STACKED_STATE_FILE,
    ServingBundle,
    ServingBundleManifest,
    bundle_directory,
    load_serving_bundle,
)
from password_attack_detector.deployment.materialize import (
    FrozenFusionEvidence,
    MaterializationOutcome,
    materialize_serving_bundle,
    read_frozen_fusion,
)

__all__ = [
    "BUNDLE_MANIFEST_FILE",
    "BUNDLE_SCHEMA_VERSION",
    "FUSION_SELECTION_FILE",
    "SERVING_BUNDLE_DIR",
    "STACKED_STATE_FILE",
    "FrozenFusionEvidence",
    "MaterializationOutcome",
    "ServingBundle",
    "ServingBundleManifest",
    "bundle_directory",
    "load_serving_bundle",
    "materialize_serving_bundle",
    "read_frozen_fusion",
]
