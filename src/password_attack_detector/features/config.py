"""Typed, versioned configuration for the feature-engineering layer.

``FeatureConfig`` is the single source of truth for one feature build.  Its
``fingerprint()`` method produces a SHA-256 hex digest covering *semantic*
feature behaviour only.  Output directories, overwrite flags, absolute paths,
creation timestamps, secrets, and machine-specific values are excluded, so the
same semantic configuration stored in two different directories fingerprints
identically.

Durations are parsed once into ``timedelta`` and carried internally as integer
microseconds.  No module in this package re-parses duration strings.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from password_attack_detector.data.enums import ScenarioType
from password_attack_detector.data.schemas import SCHEMA_VERSION
from password_attack_detector.exceptions import ConfigurationError

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "FINGERPRINT_EXCLUDED_FIELDS",
    "AggregateKind",
    "BaselineConfig",
    "EntityKind",
    "FeatureConfig",
    "GeospatialConfig",
    "SplitConfig",
    "duration_to_microseconds",
    "format_duration",
    "load_feature_config",
    "parse_duration",
]

FEATURE_SCHEMA_VERSION: str = "1.0.0"

#: Fields of :class:`FeatureConfig` deliberately excluded from its fingerprint.
#: These describe *where* output goes, not *what* the features mean.
FINGERPRINT_EXCLUDED_FIELDS: frozenset[str] = frozenset({"output_dir", "overwrite"})

_DURATION_RE = re.compile(r"^(?P<value>\d+)(?P<unit>[smhd])$")
_UNIT_SECONDS: dict[str, int] = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_MICROSECONDS_PER_SECOND: int = 1_000_000


class EntityKind(StrEnum):
    """The entity whose history a feature aggregates over."""

    USER = "user"
    SOURCE = "source"
    USER_SOURCE = "user_source"
    USER_DEVICE = "user_device"
    DEVICE = "device"
    SESSION = "session"
    NONE = "none"


class AggregateKind(StrEnum):
    """The kind of aggregation a windowed feature performs."""

    COUNT = "count"
    RATE = "rate"
    MEAN = "mean"
    STD = "std"
    COV = "cov"
    UNIQUE_COUNT = "unique_count"
    RATIO = "ratio"
    ZSCORE = "zscore"
    DELTA = "delta"
    FLAG = "flag"


#: Aggregates whose window set is restricted by ``cardinality_windows``.
_CARDINALITY_AGGREGATES: frozenset[AggregateKind] = frozenset(
    {AggregateKind.UNIQUE_COUNT}
)

#: Aggregates whose window set is restricted by ``dispersion_windows``.
_DISPERSION_AGGREGATES: frozenset[AggregateKind] = frozenset(
    {AggregateKind.STD, AggregateKind.COV}
)

#: Entity kinds whose window set is restricted by ``device_session_windows``.
_DEVICE_SESSION_ENTITIES: frozenset[EntityKind] = frozenset(
    {EntityKind.USER_DEVICE, EntityKind.DEVICE, EntityKind.SESSION}
)


def _parse_duration(value: str, *, allow_zero: bool = False) -> timedelta:
    """Parse a duration string, raising ``ValueError`` on failure.

    Pydantic wraps only ``ValueError`` and ``AssertionError`` into
    ``ValidationError``, so every validator in this module funnels through
    this helper rather than the public :func:`parse_duration`.
    """
    match = _DURATION_RE.match(value.strip())
    if match is None:
        raise ValueError(
            f"Invalid duration {value!r}; expected an integer followed by "
            f"one of s, m, h, d (for example '15m' or '24h')"
        )
    seconds = int(match.group("value")) * _UNIT_SECONDS[match.group("unit")]
    if seconds < 0 or (seconds == 0 and not allow_zero):
        raise ValueError(f"Duration {value!r} must be strictly positive")
    return timedelta(seconds=seconds)


def parse_duration(value: str) -> timedelta:
    """Parse a duration string such as ``"15m"`` or ``"24h"`` into a timedelta.

    Supported units are ``s`` (seconds), ``m`` (minutes), ``h`` (hours), and
    ``d`` (days).  The numeric part must be a positive integer.

    Raises:
        ConfigurationError: if the string is malformed or the duration is not
            strictly positive.
    """
    try:
        return _parse_duration(value)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from None


def format_duration(delta: timedelta) -> str:
    """Render a timedelta back into the canonical duration string.

    Chooses the largest unit that divides the duration exactly, so
    ``timedelta(hours=24)`` renders as ``"1d"`` only if it is a whole number of
    days *and* not expressible more naturally in hours.  Windows are rendered
    with the unit they were configured with wherever possible; this function is
    used for fingerprints and diagnostics, not for round-tripping user input.
    """
    total_seconds = int(delta.total_seconds())
    for unit in ("d", "h", "m", "s"):
        size = _UNIT_SECONDS[unit]
        if total_seconds % size == 0 and total_seconds >= size:
            return f"{total_seconds // size}{unit}"
    return f"{total_seconds}s"


def duration_to_microseconds(delta: timedelta) -> int:
    """Return the duration as an exact integer number of microseconds."""
    return int(delta.total_seconds() * _MICROSECONDS_PER_SECOND)


def _parse_duration_field(value: object) -> object:
    """Pydantic ``mode="before"`` helper accepting duration strings.

    Zero is permitted here because ``embargo: "0s"`` is a legitimate setting.
    """
    if isinstance(value, str):
        return _parse_duration(value, allow_zero=True)
    return value


def _validate_window_tuple(windows: tuple[str, ...], field_name: str) -> None:
    """Validate that *windows* is non-empty, unique, and strictly ascending.

    Raises ``ValueError`` so pydantic surfaces it as a ``ValidationError``.
    """
    if not windows:
        raise ValueError(f"{field_name} must not be empty")

    seen: set[str] = set()
    for window in windows:
        if window in seen:
            raise ValueError(f"{field_name} contains duplicate window {window!r}")
        seen.add(window)

    deltas = [_parse_duration(w) for w in windows]
    for (earlier, later), (left, right) in zip(
        pairwise(deltas), pairwise(windows), strict=True
    ):
        if later <= earlier:
            raise ValueError(
                f"{field_name} must be strictly ascending; {right!r} does not "
                f"follow {left!r}"
            )


def _validate_safe_path(path: Path | None, field_name: str) -> None:
    """Reject output paths containing parent-directory traversal."""
    if path is None:
        return
    if any(part == ".." for part in path.parts):
        raise ValueError(f"{field_name} must not contain '..' path components")


class GeospatialConfig(BaseModel):
    """Settings for coarse-location derived features.

    ``max_plausible_velocity_kmh`` is used **only** to cap and normalise the
    reported implied velocity.  It never produces an attack decision and there
    is no impossible-travel flag anywhere in this package.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    max_plausible_velocity_kmh: float = Field(default=1200.0, gt=0.0)
    min_distance_km_for_velocity: float = Field(default=0.0, ge=0.0)

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields that contribute to the config fingerprint."""
        return {
            "enabled": self.enabled,
            "max_plausible_velocity_kmh": self.max_plausible_velocity_kmh,
            "min_distance_km_for_velocity": self.min_distance_km_for_velocity,
        }


class BaselineConfig(BaseModel):
    """Settings controlling how behavioral baselines are fitted.

    ``known_set_max_size`` bounds the per-entity known-value sets so a single
    high-volume entity cannot produce an unbounded ragged Parquet cell.  Sets
    are truncated most-frequent-first and the truncation is recorded in the
    fitted state.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_schema_version: Literal["1.0.0"] = "1.0.0"
    enabled: bool = True
    min_events_per_user: int = Field(default=5, ge=1)
    min_events_per_source: int = Field(default=5, ge=1)
    known_set_max_size: int = Field(default=64, ge=1)
    known_set_min_occurrences: int = Field(default=1, ge=1)
    hour_histogram_alpha: float = Field(default=1.0, ge=0.0)
    response_time_min_events: int = Field(default=5, ge=2)
    max_response_time_zscore: float = Field(default=10.0, gt=0.0)
    max_tracked_users: int | None = Field(default=None, ge=1)
    max_tracked_sources: int | None = Field(default=None, ge=1)

    #: The feature window whose counts are compared against baseline rates.
    #: Must be one of the configured windows, and one that tracks cardinality,
    #: because the source fan-out ratio reads a unique count from it.
    rate_reference_window: str = "1h"

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields that contribute to the config fingerprint."""
        return {
            "baseline_schema_version": self.baseline_schema_version,
            "enabled": self.enabled,
            "min_events_per_user": self.min_events_per_user,
            "min_events_per_source": self.min_events_per_source,
            "known_set_max_size": self.known_set_max_size,
            "known_set_min_occurrences": self.known_set_min_occurrences,
            "hour_histogram_alpha": self.hour_histogram_alpha,
            "response_time_min_events": self.response_time_min_events,
            "max_response_time_zscore": self.max_response_time_zscore,
            "max_tracked_users": self.max_tracked_users,
            "max_tracked_sources": self.max_tracked_sources,
            "rate_reference_window": self.rate_reference_window,
        }

    def fingerprint(self) -> str:
        """Return a SHA-256 hex digest of the semantic baseline configuration."""
        canonical = json.dumps(self.fingerprint_data(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


class SplitConfig(BaseModel):
    """Settings for deterministic chronological, campaign-aware splitting.

    Features look *backwards*, so contamination flows *forward across* a
    boundary.  ``purge`` therefore excludes events on the **later** side of a
    boundary (whose lookback windows reach back across it) and ``embargo``
    excludes events on the **earlier** side.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    split_schema_version: Literal["1.0.0"] = "1.0.0"
    mode: Literal["fractions", "boundaries"] = "fractions"

    train_end: AwareDatetime | None = None
    validation_end: AwareDatetime | None = None
    test_end: AwareDatetime | None = None

    train_fraction: float = Field(default=0.70, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)

    purge: timedelta = timedelta(hours=24)
    embargo: timedelta = timedelta(0)
    strict_isolation: bool = True

    boundary_campaign_policy: Literal["exclude", "assign_by_first_event"] = "exclude"
    normal_grouping: Literal["singleton", "campaign_id"] = "singleton"
    max_excluded_fraction: float = Field(default=0.10, ge=0.0, le=1.0)

    #: Scenarios routed to the evaluation-only holdout rather than to any
    #: supervised split.  Events flagged ineligible for supervised training are
    #: routed there too, regardless of scenario.
    holdout_scenarios: frozenset[ScenarioType] = frozenset(
        {ScenarioType.NOVEL_ANOMALY_HOLDOUT}
    )

    @field_validator("purge", "embargo", mode="before")
    @classmethod
    def coerce_duration(cls, v: object) -> object:
        """Accept duration strings such as ``"24h"`` for timedelta fields."""
        return _parse_duration_field(v)

    @field_validator("purge", "embargo", mode="after")
    @classmethod
    def reject_negative_duration(cls, v: timedelta) -> timedelta:
        """Purge and embargo intervals must not be negative."""
        if v < timedelta(0):
            raise ValueError("purge and embargo must not be negative")
        return v

    @field_validator("train_end", "validation_end", "test_end", mode="after")
    @classmethod
    def normalize_to_utc(cls, v: datetime | None) -> datetime | None:
        """Normalise explicit split boundaries to UTC."""
        return None if v is None else v.astimezone(UTC)

    @model_validator(mode="after")
    def check_mode_consistency(self) -> Self:
        """Validate fractions or boundaries according to the selected mode."""
        if self.mode == "fractions":
            total = self.train_fraction + self.validation_fraction + self.test_fraction
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    f"train/validation/test fractions must sum to 1.0, got {total!r}"
                )
        else:
            boundaries = (self.train_end, self.validation_end, self.test_end)
            if any(b is None for b in boundaries):
                raise ValueError(
                    "mode='boundaries' requires train_end, validation_end, and "
                    "test_end to all be set"
                )
            ordered = [b for b in boundaries if b is not None]
            for earlier, later in pairwise(ordered):
                if later <= earlier:
                    raise ValueError(
                        "split boundaries must be strictly increasing "
                        "(train_end < validation_end < test_end)"
                    )
        return self

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields that contribute to the config fingerprint."""
        return {
            "split_schema_version": self.split_schema_version,
            "mode": self.mode,
            "train_end": None if self.train_end is None else self.train_end.isoformat(),
            "validation_end": (
                None if self.validation_end is None else self.validation_end.isoformat()
            ),
            "test_end": None if self.test_end is None else self.test_end.isoformat(),
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "purge_seconds": int(self.purge.total_seconds()),
            "embargo_seconds": int(self.embargo.total_seconds()),
            "strict_isolation": self.strict_isolation,
            "boundary_campaign_policy": self.boundary_campaign_policy,
            "normal_grouping": self.normal_grouping,
            "max_excluded_fraction": self.max_excluded_fraction,
            "holdout_scenarios": sorted(str(s) for s in self.holdout_scenarios),
        }

    def fingerprint(self) -> str:
        """Return a SHA-256 hex digest of the semantic split configuration."""
        canonical = json.dumps(self.fingerprint_data(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


class FeatureConfig(BaseModel):
    """Complete configuration for one feature build.

    The window tuples are the mechanism by which feature-set width is tuned.
    ``windows`` is the full ladder; the narrower tuples restrict specific
    entity kinds and aggregate kinds to the subset where they carry signal.
    Because the catalog expands from :meth:`windows_for`, changing these tuples
    changes the emitted schema, the Arrow types, and the validator in lockstep
    without touching code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_schema_version: Literal["1.0.0"] = "1.0.0"
    source_schema_version: str = SCHEMA_VERSION

    windows: tuple[str, ...] = ("1m", "5m", "15m", "1h", "24h")
    cardinality_windows: tuple[str, ...] = ("5m", "1h", "24h")
    dispersion_windows: tuple[str, ...] = ("5m", "15m", "1h", "24h")
    device_session_windows: tuple[str, ...] = ("5m", "1h")
    pair_windows: tuple[str, ...] = ("5m", "1h", "24h")

    anchor_mode: Literal["per_event"] = "per_event"
    null_policy: Literal["null_for_undefined"] = "null_for_undefined"
    min_count_for_rate: int = Field(default=1, ge=1)
    min_history_events: int = Field(default=0, ge=0)
    max_tracked_entities: int | None = Field(default=None, ge=1)

    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    geospatial: GeospatialConfig = Field(default_factory=GeospatialConfig)

    # Excluded from the fingerprint by construction -- see fingerprint().
    output_dir: Path | None = None
    overwrite: bool = False

    @field_validator(
        "windows",
        "cardinality_windows",
        "dispersion_windows",
        "device_session_windows",
        "pair_windows",
        mode="after",
    )
    @classmethod
    def check_window_tuple(cls, v: tuple[str, ...], info: Any) -> tuple[str, ...]:
        """Reject empty, duplicated, malformed, or non-ascending window tuples."""
        _validate_window_tuple(v, str(info.field_name))
        return v

    @field_validator("output_dir", mode="after")
    @classmethod
    def check_output_dir(cls, v: Path | None) -> Path | None:
        """Reject output directories containing parent-directory traversal."""
        _validate_safe_path(v, "output_dir")
        return v

    @model_validator(mode="after")
    def check_cross_field_constraints(self) -> Self:
        """Validate subset relationships, schema compatibility, and purge length."""
        if self.source_schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"source_schema_version {self.source_schema_version!r} is "
                f"incompatible with the canonical event schema version "
                f"{SCHEMA_VERSION!r}"
            )

        base = set(self.windows)
        for name in (
            "cardinality_windows",
            "dispersion_windows",
            "device_session_windows",
            "pair_windows",
        ):
            subset: tuple[str, ...] = getattr(self, name)
            extra = sorted(set(subset) - base)
            if extra:
                raise ValueError(
                    f"{name} contains window(s) {extra} not present in `windows`"
                )

        reference = self.baseline.rate_reference_window
        if self.baseline.enabled and reference not in self.windows:
            raise ValueError(
                f"baseline.rate_reference_window {reference!r} is not one of the "
                f"configured windows {list(self.windows)}"
            )
        if self.baseline.enabled and reference not in self.cardinality_windows:
            raise ValueError(
                f"baseline.rate_reference_window {reference!r} must also appear "
                f"in cardinality_windows; the source fan-out ratio reads a "
                f"unique count from it"
            )

        if self.split.strict_isolation and self.split.purge < self.max_window:
            raise ValueError(
                f"split.purge ({format_duration(self.split.purge)}) is shorter "
                f"than the maximum feature window "
                f"({format_duration(self.max_window)}); either raise purge or "
                f"set split.strict_isolation=false"
            )

        return self

    @property
    def max_window(self) -> timedelta:
        """Return the longest configured feature window."""
        return max(_parse_duration(w) for w in self.windows)

    @property
    def max_window_microseconds(self) -> int:
        """Return the longest configured feature window in integer microseconds."""
        return duration_to_microseconds(self.max_window)

    def window_microseconds(self) -> tuple[tuple[str, int], ...]:
        """Return ``(label, microseconds)`` for every configured window, ascending."""
        return tuple(
            (w, duration_to_microseconds(_parse_duration(w))) for w in self.windows
        )

    def windows_for(
        self, kind: EntityKind, aggregate: AggregateKind
    ) -> tuple[str, ...]:
        """Return the windows at which ``(kind, aggregate)`` features are emitted.

        The result preserves the ascending order of :attr:`windows`.  This is
        the single place where feature-width trims are applied; the catalog,
        the Arrow schema, and the validator all derive from it.
        """
        if kind in _DEVICE_SESSION_ENTITIES:
            allowed = set(self.device_session_windows)
        elif kind is EntityKind.USER_SOURCE:
            allowed = set(self.pair_windows)
        else:
            allowed = set(self.windows)

        if aggregate in _CARDINALITY_AGGREGATES:
            allowed &= set(self.cardinality_windows)
        elif aggregate in _DISPERSION_AGGREGATES:
            allowed &= set(self.dispersion_windows)

        return tuple(w for w in self.windows if w in allowed)

    def fingerprint_data(self) -> dict[str, Any]:
        """Return the semantic fields that contribute to the config fingerprint.

        Every field of this model must appear here except those listed in
        :data:`FINGERPRINT_EXCLUDED_FIELDS`.  A test asserts that invariant, so
        a field added to the model without a decision about its fingerprint
        status fails the build rather than silently weakening the digest.
        """
        data: dict[str, Any] = {
            "feature_schema_version": self.feature_schema_version,
            "source_schema_version": self.source_schema_version,
            "windows": list(self.windows),
            "cardinality_windows": list(self.cardinality_windows),
            "dispersion_windows": list(self.dispersion_windows),
            "device_session_windows": list(self.device_session_windows),
            "pair_windows": list(self.pair_windows),
            "anchor_mode": self.anchor_mode,
            "null_policy": self.null_policy,
            "min_count_for_rate": self.min_count_for_rate,
            "min_history_events": self.min_history_events,
            "max_tracked_entities": self.max_tracked_entities,
            "baseline": self.baseline.fingerprint_data(),
            "split": self.split.fingerprint_data(),
            "geospatial": self.geospatial.fingerprint_data(),
        }
        return data

    def fingerprint(self) -> str:
        """Return a SHA-256 hex digest of the semantic feature configuration.

        The digest covers schema versions, every window tuple, anchor and null
        policy, rate and history thresholds, the entity cap, and the nested
        baseline, split, and geospatial configurations.

        It deliberately **excludes** ``output_dir``, ``overwrite``, absolute
        paths, creation timestamps, secrets, and machine-specific values, so
        the same semantic configuration in different directories produces the
        same fingerprint.
        """
        canonical = json.dumps(self.fingerprint_data(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_feature_config(path: Path) -> FeatureConfig:
    """Load and validate a :class:`FeatureConfig` from a YAML file.

    Raises:
        ConfigurationError: if the file is missing, unreadable, not a mapping,
            or fails validation.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read feature configuration: {type(exc).__name__}"
        ) from None

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        raise ConfigurationError("Feature configuration is not valid YAML") from None

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigurationError("Feature configuration must be a YAML mapping")

    try:
        return FeatureConfig(**data)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Invalid feature configuration: {exc}") from None
