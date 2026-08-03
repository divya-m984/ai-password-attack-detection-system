"""Point-in-time feature engineering and behavioral baselines.

This package converts validated canonical authentication events into a
model-ready feature layer.  It trains no model, makes no detection decision,
and produces no risk score.

The central guarantee is the *point-in-time contract*: every historical
aggregate for an anchor event at time ``t`` is computed from events in the
half-open interval ``[t - window, t)`` only.  The anchor never enters its own
history, and events sharing the anchor's exact timestamp never enter it
either.  See ``docs/temporal-semantics.md``.
"""

from __future__ import annotations

from password_attack_detector.features.config import (
    FEATURE_SCHEMA_VERSION,
    AggregateKind,
    BaselineConfig,
    EntityKind,
    FeatureConfig,
    GeospatialConfig,
    SplitConfig,
    load_feature_config,
    parse_duration,
)

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "AggregateKind",
    "BaselineConfig",
    "EntityKind",
    "FeatureConfig",
    "GeospatialConfig",
    "SplitConfig",
    "load_feature_config",
    "parse_duration",
]
