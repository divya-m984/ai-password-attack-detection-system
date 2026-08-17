"""The HTTP serving layer: an adapter around the frozen detection system.

Nothing in this package decides anything scientific.  The rules, the feature
computation, the champion model, its calibrator, its operating point, and the
fusion strategy were all chosen and frozen in Phases 3-5; this package resolves
where those artifacts live, verifies them at startup, converts an HTTP request
into the canonical authentication events the existing pipeline already accepts,
and renders what the pipeline said back as JSON.

Three properties are structural rather than conventional:

* **No second implementation.**  Features come from
  :class:`~password_attack_detector.features.engine.FeatureEngine`, rule verdicts
  from :class:`~password_attack_detector.detection.engine.DetectionEngine` and
  :class:`~password_attack_detector.detection.scoring.RiskScorer`, model scores
  from :func:`~password_attack_detector.ml.predictions.predict_binary`, and the
  hybrid verdict from :func:`~password_attack_detector.ml.fusion.fuse`.  There is
  no arithmetic in this package that produces a detection quantity.
* **No scientific override reaches the wire.**  A request cannot name a model, a
  threshold, a fusion strategy, an artifact path, or a feature; the request
  schemas forbid extra fields and
  :func:`~password_attack_detector.api.config.load_api_settings` declares no
  field through which one could arrive.
* **No credential material is accepted.**  Every request model rejects a field
  whose name normalises to a prohibited credential term, before any other
  validation runs.
"""

from __future__ import annotations

from password_attack_detector.api.config import APISettings, load_api_settings
from password_attack_detector.api.errors import APIError, ErrorCode
from password_attack_detector.api.schemas import API_SCHEMA_VERSION

__all__ = [
    "API_SCHEMA_VERSION",
    "APIError",
    "APISettings",
    "ErrorCode",
    "load_api_settings",
]
