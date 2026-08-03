"""The versioned feature catalog -- the single source of truth for the schema.

Every emitted column is described by exactly one :class:`FeatureSpec`.  The
catalog drives:

* the Arrow schema used to write ``feature_snapshots.parquet``;
* the column projection applied to engine output (a missing or undeclared key
  raises rather than silently changing the schema);
* the empty snapshot used for entities with no history;
* the dataset validator's type, nullability, and range checks;
* the leakage auditor's classification of prior-history versus current-event
  columns.

Because the catalog expands from :meth:`FeatureConfig.windows_for`, feature
width is tuned by configuration rather than by editing code.

There is deliberately no parallel pydantic row model.  A second declaration of
~200 columns would be a second source of truth, and it would drift.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from password_attack_detector.data.schemas import PROHIBITED_GT_COLUMNS
from password_attack_detector.exceptions import FeatureComputationError
from password_attack_detector.features.config import (
    FEATURE_SCHEMA_VERSION,
    AggregateKind,
    EntityKind,
    FeatureConfig,
)

__all__ = [
    "AMBIGUOUS_FEATURE_NAMES",
    "ANCHOR_EVENT_ID",
    "ANCHOR_EVENT_TIME",
    "PROHIBITED_FEATURE_COLUMNS",
    "FeatureCatalog",
    "FeatureDType",
    "FeatureGroup",
    "FeatureSpec",
    "LeakageClass",
    "build_catalog",
    "catalog_to_markdown",
]

ANCHOR_EVENT_ID: str = "anchor_event_id"
ANCHOR_EVENT_TIME: str = "anchor_event_time"

#: Names that assert a conclusion rather than describe an observation, or that
#: are too vague to be machine-readable.  Rejected at catalog build time.
AMBIGUOUS_FEATURE_NAMES: frozenset[str] = frozenset(
    {
        "attempts",
        "score",
        "suspicious",
        "anomaly",
        "anomalous",
        "attack",
        "threat",
        "bot_detected",
        "account_takeover",
        "password_spraying",
        "brute_force",
        "impossible_travel",
        "impossible_travel_attack",
        "credential_stuffing",
    }
)

#: Columns that must never appear in the feature matrix: ground truth, campaign
#: metadata, split assignment, and model output.
PROHIBITED_FEATURE_COLUMNS: frozenset[str] = PROHIBITED_GT_COLUMNS | frozenset(
    {
        "split",
        "exclusion_reason",
        "campaign_stage",
        "scenario_variant",
        "attack_class",
        "model_probability",
    }
)


class FeatureGroup(StrEnum):
    """Coarse grouping used for reporting and later feature selection."""

    KEY = "key"
    CURRENT_CONTEXT = "current_context"
    CALENDAR = "calendar"
    USER_HISTORY = "user_history"
    SOURCE_HISTORY = "source_history"
    PAIR_HISTORY = "pair_history"
    DEVICE_SESSION = "device_session"
    SEQUENCE = "sequence"
    GEOSPATIAL = "geospatial"
    BASELINE = "baseline"


class LeakageClass(StrEnum):
    """How a feature relates to the anchor event in time.

    This field is load-bearing: the leakage auditor uses it to decide which
    columns must be invariant under future mutation, rather than pattern
    matching on column names.
    """

    #: Computed strictly from events in ``[t - window, t)``.
    PRIOR_ONLY = "prior_only"
    #: The anchor event's own recorded fields.  Legitimate because detection
    #: happens after the event has completed.
    CURRENT_EVENT_CONTEXT = "current_event_context"
    #: Derived from a baseline fitted on an approved reference interval.
    BASELINE_DERIVED = "baseline_derived"
    #: Identity and timing keys.
    KEY = "key"


class FeatureDType(StrEnum):
    """Logical storage type of a feature column."""

    INT64 = "int64"
    FLOAT64 = "float64"
    BOOL = "bool"
    STRING = "string"
    TIMESTAMP = "timestamp"


class FeatureSpec(BaseModel):
    """Complete, machine-readable description of one feature column."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    group: FeatureGroup
    entity: EntityKind
    window: str | None = None
    aggregate: AggregateKind | None = None
    dtype: FeatureDType
    nullable: bool
    unit: str | None = None
    leakage_class: LeakageClass
    min_count: int | None = None
    min_observations: int | None = None
    value_range: tuple[float, float] | None = None
    null_semantics: str
    description: str
    privacy_class: Literal["non_sensitive", "operational_metadata"] = "non_sensitive"
    intended_use: str = "supervised classification and anomaly detection"
    requires_baseline: bool = False
    uses_current_event: bool = False
    deprecated: bool = False
    companion_of: str | None = None

    @property
    def is_count_like(self) -> bool:
        """True when the feature is a non-negative integer tally."""
        return self.aggregate in {AggregateKind.COUNT, AggregateKind.UNIQUE_COUNT}


#: Fields that participate in the catalog fingerprint.  ``description`` and
#: ``null_semantics`` are excluded on purpose: fixing a typo in prose must not
#: invalidate every downstream artifact that records the fingerprint.
_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "name",
    "group",
    "entity",
    "window",
    "aggregate",
    "dtype",
    "nullable",
    "unit",
    "leakage_class",
    "min_count",
    "min_observations",
    "value_range",
    "privacy_class",
    "requires_baseline",
    "uses_current_event",
    "deprecated",
    "companion_of",
)


@dataclass(frozen=True, slots=True)
class _WindowedTemplate:
    """A windowed feature, expanded once per configured window."""

    measure: str
    aggregate: AggregateKind
    dtype: FeatureDType
    nullable: bool
    unit: str | None
    null_semantics: str
    description: str
    value_range: tuple[float, float] | None = None
    min_observations: int | None = None


_ENTITY_PREFIX: dict[EntityKind, str] = {
    EntityKind.USER: "user",
    EntityKind.SOURCE: "source",
    EntityKind.USER_SOURCE: "pair",
    EntityKind.USER_DEVICE: "user_device",
    EntityKind.DEVICE: "device",
    EntityKind.SESSION: "session",
}

_ENTITY_GROUP: dict[EntityKind, FeatureGroup] = {
    EntityKind.USER: FeatureGroup.USER_HISTORY,
    EntityKind.SOURCE: FeatureGroup.SOURCE_HISTORY,
    EntityKind.USER_SOURCE: FeatureGroup.PAIR_HISTORY,
    EntityKind.USER_DEVICE: FeatureGroup.DEVICE_SESSION,
    EntityKind.DEVICE: FeatureGroup.DEVICE_SESSION,
    EntityKind.SESSION: FeatureGroup.DEVICE_SESSION,
}

_COUNT_NULL = "Never null; zero means no qualifying events in the window."
_RATE_NULL = (
    "Null when the denominator is below min_count (no observed attempts in the "
    "window); never 0.0 for an empty denominator."
)
_MEAN_NULL = "Null when the window contains no qualifying observations."
_STD_NULL = (
    "Null when the window contains fewer than two qualifying observations; "
    "never 0.0 for a single observation."
)
_COV_NULL = (
    "Null when the window contains fewer than two qualifying observations or "
    "the mean is zero."
)


def _count(measure: str, description: str) -> _WindowedTemplate:
    return _WindowedTemplate(
        measure=measure,
        aggregate=AggregateKind.COUNT,
        dtype=FeatureDType.INT64,
        nullable=False,
        unit="count",
        null_semantics=_COUNT_NULL,
        description=description,
        value_range=(0.0, float("inf")),
    )


def _unique(measure: str, description: str) -> _WindowedTemplate:
    return _WindowedTemplate(
        measure=measure,
        aggregate=AggregateKind.UNIQUE_COUNT,
        dtype=FeatureDType.INT64,
        nullable=False,
        unit="count",
        null_semantics=_COUNT_NULL,
        description=description,
        value_range=(0.0, float("inf")),
    )


def _rate(measure: str, description: str) -> _WindowedTemplate:
    return _WindowedTemplate(
        measure=measure,
        aggregate=AggregateKind.RATE,
        dtype=FeatureDType.FLOAT64,
        nullable=True,
        unit="ratio",
        null_semantics=_RATE_NULL,
        description=description,
        value_range=(0.0, 1.0),
    )


def _mean(measure: str, unit: str, description: str) -> _WindowedTemplate:
    return _WindowedTemplate(
        measure=measure,
        aggregate=AggregateKind.MEAN,
        dtype=FeatureDType.FLOAT64,
        nullable=True,
        unit=unit,
        null_semantics=_MEAN_NULL,
        description=description,
        value_range=(0.0, float("inf")),
        min_observations=1,
    )


def _std(measure: str, unit: str, description: str) -> _WindowedTemplate:
    return _WindowedTemplate(
        measure=measure,
        aggregate=AggregateKind.STD,
        dtype=FeatureDType.FLOAT64,
        nullable=True,
        unit=unit,
        null_semantics=_STD_NULL,
        description=description,
        value_range=(0.0, float("inf")),
        min_observations=2,
    )


_USER_TEMPLATES: tuple[_WindowedTemplate, ...] = (
    _count("attempt_count", "Authentication events recorded for this user."),
    _count("success_count", "Successful authentications for this user."),
    _count("failure_count", "Failed authentications for this user."),
    _count("challenge_count", "Challenged authentications for this user."),
    _count("blocked_count", "Blocked authentications for this user."),
    _count("mfa_failure_count", "Events for this user whose MFA outcome failed."),
    _rate("failure_rate", "Share of this user's attempts that failed."),
    _rate("success_rate", "Share of this user's attempts that succeeded."),
    _unique("unique_source_count", "Distinct sources used by this user."),
    _unique("unique_device_count", "Distinct devices used by this user."),
    _unique("unique_country_count", "Distinct countries seen for this user."),
    _unique("unique_application_count", "Distinct applications this user reached."),
    _unique("unique_auth_method_count", "Distinct authentication methods used."),
    _mean(
        "mean_response_time_ms",
        "milliseconds",
        "Mean recorded response time across this user's events.",
    ),
    _std(
        "response_time_std_ms",
        "milliseconds",
        "Sample standard deviation of this user's response times.",
    ),
    _mean(
        "mean_interarrival_seconds",
        "seconds",
        "Mean gap between this user's consecutive in-window events.",
    ),
    _std(
        "interarrival_std_seconds",
        "seconds",
        "Sample standard deviation of this user's in-window event gaps.",
    ),
)

_SOURCE_TEMPLATES: tuple[_WindowedTemplate, ...] = (
    _count("attempt_count", "Authentication events recorded from this source."),
    _count("success_count", "Successful authentications from this source."),
    _count("failure_count", "Failed authentications from this source."),
    _rate("failure_rate", "Share of this source's attempts that failed."),
    _unique("unique_user_count", "Distinct users targeted from this source."),
    _unique("unique_device_count", "Distinct devices seen from this source."),
    _unique("unique_country_count", "Distinct countries seen from this source."),
    _unique("unique_application_count", "Distinct applications reached."),
    _unique("unique_user_agent_count", "Distinct user-agent families seen."),
    _mean(
        "mean_response_time_ms",
        "milliseconds",
        "Mean recorded response time across this source's events.",
    ),
    _mean(
        "mean_interarrival_seconds",
        "seconds",
        "Mean gap between this source's consecutive in-window events.",
    ),
    _std(
        "interarrival_std_seconds",
        "seconds",
        "Sample standard deviation of this source's in-window event gaps.",
    ),
    _WindowedTemplate(
        measure="interarrival_coefficient_of_variation",
        aggregate=AggregateKind.COV,
        dtype=FeatureDType.FLOAT64,
        nullable=True,
        unit="ratio",
        null_semantics=_COV_NULL,
        description=(
            "Interarrival standard deviation divided by the interarrival mean. "
            "Low values indicate highly regular timing."
        ),
        value_range=(0.0, float("inf")),
        min_observations=2,
    ),
)

_PAIR_TEMPLATES: tuple[_WindowedTemplate, ...] = (
    _count("attempt_count", "Events for this user and source together."),
    _count("success_count", "Successful events for this user-source pair."),
    _count("failure_count", "Failed events for this user-source pair."),
    _rate("failure_rate", "Share of this pair's attempts that failed."),
    _mean(
        "mean_interarrival_seconds",
        "seconds",
        "Mean gap between this pair's consecutive in-window events.",
    ),
)

_USER_DEVICE_TEMPLATES: tuple[_WindowedTemplate, ...] = (
    _count("attempt_count", "Events for this user on this device."),
)

_SESSION_TEMPLATES: tuple[_WindowedTemplate, ...] = (
    _count("event_count", "Events recorded within this session."),
    _count("failure_count", "Failed events within this session."),
    _count("success_count", "Successful events within this session."),
)

_DEVICE_TEMPLATES: tuple[_WindowedTemplate, ...] = (
    _unique("user_count", "Distinct users seen on this device."),
)

_TEMPLATE_MATRIX: tuple[tuple[EntityKind, tuple[_WindowedTemplate, ...]], ...] = (
    (EntityKind.USER, _USER_TEMPLATES),
    (EntityKind.SOURCE, _SOURCE_TEMPLATES),
    (EntityKind.USER_SOURCE, _PAIR_TEMPLATES),
    (EntityKind.USER_DEVICE, _USER_DEVICE_TEMPLATES),
    (EntityKind.SESSION, _SESSION_TEMPLATES),
    (EntityKind.DEVICE, _DEVICE_TEMPLATES),
)


def _expand(
    template: _WindowedTemplate,
    kind: EntityKind,
    window: str,
    config: FeatureConfig,
) -> FeatureSpec:
    """Expand one template at one window into a concrete :class:`FeatureSpec`."""
    return FeatureSpec(
        name=f"{_ENTITY_PREFIX[kind]}_{template.measure}__{window}",
        group=_ENTITY_GROUP[kind],
        entity=kind,
        window=window,
        aggregate=template.aggregate,
        dtype=template.dtype,
        nullable=template.nullable,
        unit=template.unit,
        leakage_class=LeakageClass.PRIOR_ONLY,
        min_count=(
            config.min_count_for_rate
            if template.aggregate is AggregateKind.RATE
            else None
        ),
        min_observations=template.min_observations,
        value_range=template.value_range,
        null_semantics=template.null_semantics,
        description=(
            f"{template.description} Computed over the half-open interval "
            f"[t - {window}, t); the anchor event and any event sharing its "
            f"timestamp are excluded."
        ),
    )


def _key_specs() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(
            name="feature_schema_version",
            group=FeatureGroup.KEY,
            entity=EntityKind.NONE,
            dtype=FeatureDType.STRING,
            nullable=False,
            leakage_class=LeakageClass.KEY,
            null_semantics="Never null.",
            description="Version of the feature schema this row conforms to.",
            intended_use="schema compatibility checking",
        ),
        FeatureSpec(
            name=ANCHOR_EVENT_ID,
            group=FeatureGroup.KEY,
            entity=EntityKind.NONE,
            dtype=FeatureDType.STRING,
            nullable=False,
            leakage_class=LeakageClass.KEY,
            null_semantics="Never null.",
            description=(
                "Identifier of the canonical authentication event this snapshot "
                "describes. The only join key to the label and split tables."
            ),
            intended_use="join key; never a model input",
        ),
        FeatureSpec(
            name=ANCHOR_EVENT_TIME,
            group=FeatureGroup.KEY,
            entity=EntityKind.NONE,
            dtype=FeatureDType.TIMESTAMP,
            nullable=False,
            unit="utc timestamp",
            leakage_class=LeakageClass.KEY,
            null_semantics="Never null.",
            description="UTC time of the anchor event.",
            intended_use="chronological ordering and splitting",
        ),
    )


def _current_context_specs() -> tuple[FeatureSpec, ...]:
    """The anchor event's own recorded fields.

    These are legitimate inputs because detection runs after the event has
    completed and been recorded.  They are all classified
    ``CURRENT_EVENT_CONTEXT`` so the leakage auditor can distinguish them from
    prior-history features.
    """

    def spec(
        name: str,
        dtype: FeatureDType,
        nullable: bool,
        description: str,
        *,
        unit: str | None = None,
        null_semantics: str = "Null when the source event did not record it.",
        value_range: tuple[float, float] | None = None,
    ) -> FeatureSpec:
        return FeatureSpec(
            name=name,
            group=FeatureGroup.CURRENT_CONTEXT,
            entity=EntityKind.NONE,
            dtype=dtype,
            nullable=nullable,
            unit=unit,
            leakage_class=LeakageClass.CURRENT_EVENT_CONTEXT,
            null_semantics=null_semantics,
            description=description,
            uses_current_event=True,
            value_range=value_range,
        )

    return (
        spec(
            "current_authentication_outcome",
            FeatureDType.STRING,
            False,
            "Recorded outcome of the anchor event.",
            null_semantics="Never null.",
        ),
        spec(
            "current_authentication_method",
            FeatureDType.STRING,
            False,
            "Authentication method used by the anchor event.",
            null_semantics="Never null.",
        ),
        spec(
            "current_mfa_outcome",
            FeatureDType.STRING,
            True,
            "MFA outcome recorded for the anchor event.",
        ),
        spec(
            "current_client_type",
            FeatureDType.STRING,
            True,
            "Client type recorded for the anchor event.",
        ),
        spec(
            "current_response_time_ms",
            FeatureDType.INT64,
            True,
            "Response time recorded for the anchor event.",
            unit="milliseconds",
            value_range=(0.0, 30000.0),
        ),
        spec(
            "current_country_code",
            FeatureDType.STRING,
            True,
            "ISO 3166-1 alpha-2 country code of the anchor event.",
        ),
        spec(
            "current_has_device",
            FeatureDType.BOOL,
            False,
            "Whether the anchor event carries a device identifier.",
            null_semantics="Never null.",
        ),
        spec(
            "current_has_location",
            FeatureDType.BOOL,
            False,
            "Whether the anchor event carries usable coarse coordinates.",
            null_semantics="Never null.",
        ),
    )


def _calendar_specs() -> tuple[FeatureSpec, ...]:
    """Calendar features derived from the anchor timestamp, always in UTC.

    A user's local timezone is never inferred from country.
    """

    def cyclical(name: str, description: str) -> FeatureSpec:
        return FeatureSpec(
            name=name,
            group=FeatureGroup.CALENDAR,
            entity=EntityKind.NONE,
            dtype=FeatureDType.FLOAT64,
            nullable=False,
            unit="unitless",
            leakage_class=LeakageClass.CURRENT_EVENT_CONTEXT,
            value_range=(-1.0, 1.0),
            null_semantics="Never null.",
            description=description,
            uses_current_event=True,
        )

    return (
        FeatureSpec(
            name="hour_of_day",
            group=FeatureGroup.CALENDAR,
            entity=EntityKind.NONE,
            dtype=FeatureDType.INT64,
            nullable=False,
            unit="hour",
            leakage_class=LeakageClass.CURRENT_EVENT_CONTEXT,
            value_range=(0.0, 23.0),
            null_semantics="Never null.",
            description="UTC hour of the anchor event.",
            uses_current_event=True,
        ),
        FeatureSpec(
            name="day_of_week",
            group=FeatureGroup.CALENDAR,
            entity=EntityKind.NONE,
            dtype=FeatureDType.INT64,
            nullable=False,
            unit="day",
            leakage_class=LeakageClass.CURRENT_EVENT_CONTEXT,
            value_range=(0.0, 6.0),
            null_semantics="Never null.",
            description="UTC day of week of the anchor event; Monday is 0.",
            uses_current_event=True,
        ),
        FeatureSpec(
            name="is_weekend",
            group=FeatureGroup.CALENDAR,
            entity=EntityKind.NONE,
            dtype=FeatureDType.BOOL,
            nullable=False,
            leakage_class=LeakageClass.CURRENT_EVENT_CONTEXT,
            null_semantics="Never null.",
            description="Whether the anchor event falls on a Saturday or Sunday in UTC.",
            uses_current_event=True,
        ),
        cyclical("hour_sin", "Sine encoding of the UTC hour, period 24."),
        cyclical("hour_cos", "Cosine encoding of the UTC hour, period 24."),
        cyclical("day_of_week_sin", "Sine encoding of the UTC weekday, period 7."),
        cyclical("day_of_week_cos", "Cosine encoding of the UTC weekday, period 7."),
    )


def _sequence_specs() -> tuple[FeatureSpec, ...]:
    """Stateful features referring strictly to events before the anchor.

    A field named ``prior_*`` never observes the anchor's own outcome; the
    anchor updates sequence state only after its row has been emitted.
    """

    def counter(name: str, entity: EntityKind, description: str) -> FeatureSpec:
        return FeatureSpec(
            name=name,
            group=FeatureGroup.SEQUENCE,
            entity=entity,
            aggregate=AggregateKind.COUNT,
            dtype=FeatureDType.INT64,
            nullable=False,
            unit="count",
            leakage_class=LeakageClass.PRIOR_ONLY,
            value_range=(0.0, float("inf")),
            null_semantics="Never null; zero means no such prior run exists.",
            description=description,
        )

    def elapsed(name: str, entity: EntityKind, description: str) -> FeatureSpec:
        return FeatureSpec(
            name=name,
            group=FeatureGroup.SEQUENCE,
            entity=entity,
            aggregate=AggregateKind.DELTA,
            dtype=FeatureDType.FLOAT64,
            nullable=True,
            unit="seconds",
            leakage_class=LeakageClass.PRIOR_ONLY,
            value_range=(0.0, float("inf")),
            null_semantics="Null when no qualifying prior event exists.",
            description=description,
        )

    def outcome(name: str, entity: EntityKind, description: str) -> FeatureSpec:
        return FeatureSpec(
            name=name,
            group=FeatureGroup.SEQUENCE,
            entity=entity,
            dtype=FeatureDType.STRING,
            nullable=True,
            leakage_class=LeakageClass.PRIOR_ONLY,
            null_semantics="Null when no prior event exists for this entity.",
            description=description,
        )

    return (
        counter(
            "prior_consecutive_user_failures",
            EntityKind.USER,
            "Length of the unbroken run of failures immediately before the anchor.",
        ),
        counter(
            "prior_consecutive_source_failures",
            EntityKind.SOURCE,
            "Length of the unbroken run of failures from this source before the anchor.",
        ),
        counter(
            "prior_failures_since_user_success",
            EntityKind.USER,
            "Failures recorded for this user since their last success.",
        ),
        counter(
            "prior_failures_since_pair_success",
            EntityKind.USER_SOURCE,
            "Failures for this user-source pair since the pair's last success.",
        ),
        elapsed(
            "seconds_since_user_previous_event",
            EntityKind.USER,
            "Elapsed time since this user's previous event.",
        ),
        elapsed(
            "seconds_since_source_previous_event",
            EntityKind.SOURCE,
            "Elapsed time since this source's previous event.",
        ),
        elapsed(
            "seconds_since_user_previous_success",
            EntityKind.USER,
            "Elapsed time since this user's previous successful authentication.",
        ),
        elapsed(
            "seconds_since_user_previous_failure",
            EntityKind.USER,
            "Elapsed time since this user's previous failed authentication.",
        ),
        elapsed(
            "seconds_since_pair_previous_event",
            EntityKind.USER_SOURCE,
            "Elapsed time since this user-source pair's previous event.",
        ),
        outcome(
            "previous_user_outcome",
            EntityKind.USER,
            "Outcome of this user's previous event; never the anchor's own outcome.",
        ),
        outcome(
            "previous_pair_outcome",
            EntityKind.USER_SOURCE,
            "Outcome of this pair's previous event; never the anchor's own outcome.",
        ),
    )


def _geospatial_specs() -> tuple[FeatureSpec, ...]:
    """Coarse-location derived features.

    Null has several distinct causes here, so the family carries two
    categorical status columns rather than a boolean validity twin per feature.
    No column asserts an attack: there is no impossible-travel flag.
    """
    return (
        FeatureSpec(
            name="distance_km_from_user_previous_success",
            group=FeatureGroup.GEOSPATIAL,
            entity=EntityKind.USER,
            aggregate=AggregateKind.DELTA,
            dtype=FeatureDType.FLOAT64,
            nullable=True,
            unit="km",
            leakage_class=LeakageClass.PRIOR_ONLY,
            value_range=(0.0, 20100.0),
            null_semantics=(
                "Null unless user_previous_success_geo__status is 'ok'; see that "
                "column for the reason."
            ),
            description=(
                "Great-circle distance between the anchor's coarse coordinates "
                "and those of this user's previous successful authentication."
            ),
        ),
        FeatureSpec(
            name="seconds_since_user_previous_success_with_location",
            group=FeatureGroup.GEOSPATIAL,
            entity=EntityKind.USER,
            aggregate=AggregateKind.DELTA,
            dtype=FeatureDType.FLOAT64,
            nullable=True,
            unit="seconds",
            leakage_class=LeakageClass.PRIOR_ONLY,
            value_range=(0.0, float("inf")),
            null_semantics="Null when no prior located success exists for this user.",
            description=(
                "Elapsed time since this user's previous successful authentication "
                "that carried usable coordinates."
            ),
        ),
        FeatureSpec(
            name="implied_velocity_kmh_from_previous_success",
            group=FeatureGroup.GEOSPATIAL,
            entity=EntityKind.USER,
            aggregate=AggregateKind.RATIO,
            dtype=FeatureDType.FLOAT64,
            nullable=True,
            unit="kmh",
            leakage_class=LeakageClass.PRIOR_ONLY,
            value_range=(0.0, float("inf")),
            null_semantics=(
                "Null unless implied_velocity__status is 'ok' or 'capped'; never "
                "computed when the elapsed interval is zero."
            ),
            description=(
                "Distance divided by elapsed time since the previous located "
                "success, capped at the configured plausible maximum. This is a "
                "normalised magnitude, not a travel-feasibility decision."
            ),
        ),
        FeatureSpec(
            name="country_changed_since_previous_success",
            group=FeatureGroup.GEOSPATIAL,
            entity=EntityKind.USER,
            aggregate=AggregateKind.FLAG,
            dtype=FeatureDType.BOOL,
            nullable=True,
            leakage_class=LeakageClass.PRIOR_ONLY,
            null_semantics="Null when either country code is unavailable.",
            description=(
                "Whether the anchor's country differs from that of this user's "
                "previous successful authentication."
            ),
        ),
        FeatureSpec(
            name="region_changed_since_previous_success",
            group=FeatureGroup.GEOSPATIAL,
            entity=EntityKind.USER,
            aggregate=AggregateKind.FLAG,
            dtype=FeatureDType.BOOL,
            nullable=True,
            leakage_class=LeakageClass.PRIOR_ONLY,
            null_semantics="Null when either region code is unavailable.",
            description=(
                "Whether the anchor's region differs from that of this user's "
                "previous successful authentication."
            ),
        ),
        FeatureSpec(
            name="user_previous_success_geo__status",
            group=FeatureGroup.GEOSPATIAL,
            entity=EntityKind.USER,
            dtype=FeatureDType.STRING,
            nullable=False,
            leakage_class=LeakageClass.PRIOR_ONLY,
            null_semantics="Never null.",
            description=(
                "Why the previous-success distance is or is not available: one of "
                "'ok', 'no_prior_success', 'missing_current_location', "
                "'missing_prior_location'."
            ),
            companion_of="distance_km_from_user_previous_success",
        ),
        FeatureSpec(
            name="implied_velocity__status",
            group=FeatureGroup.GEOSPATIAL,
            entity=EntityKind.USER,
            dtype=FeatureDType.STRING,
            nullable=False,
            leakage_class=LeakageClass.PRIOR_ONLY,
            null_semantics="Never null.",
            description=(
                "Why the implied velocity is or is not available: one of 'ok', "
                "'zero_elapsed', 'capped', 'unavailable'."
            ),
            companion_of="implied_velocity_kmh_from_previous_success",
        ),
    )


def _baseline_specs(config: FeatureConfig) -> tuple[FeatureSpec, ...]:
    """Features derived from a fitted behavioral baseline.

    All of these are null when the entity is absent from the fitted baseline.
    A cold user must not be reported as having a *new* device: that would
    conflate "never seen this user" with "know this user, and this device is
    new", which is exactly the encoding a model could use to recover the split
    boundary.  The two coverage columns make the distinction explicit.
    """
    absent = (
        "Null when the entity is absent from the fitted baseline; see "
        "user_in_baseline / source_in_baseline."
    )

    def novelty(name: str, description: str) -> FeatureSpec:
        return FeatureSpec(
            name=name,
            group=FeatureGroup.BASELINE,
            entity=EntityKind.USER,
            aggregate=AggregateKind.FLAG,
            dtype=FeatureDType.BOOL,
            nullable=True,
            leakage_class=LeakageClass.BASELINE_DERIVED,
            min_observations=config.baseline.min_events_per_user,
            null_semantics=absent,
            description=description,
            requires_baseline=True,
        )

    def numeric(
        name: str,
        entity: EntityKind,
        aggregate: AggregateKind,
        unit: str,
        description: str,
        *,
        value_range: tuple[float, float] | None = None,
    ) -> FeatureSpec:
        return FeatureSpec(
            name=name,
            group=FeatureGroup.BASELINE,
            entity=entity,
            aggregate=aggregate,
            dtype=FeatureDType.FLOAT64,
            nullable=True,
            unit=unit,
            leakage_class=LeakageClass.BASELINE_DERIVED,
            min_observations=(
                config.baseline.min_events_per_user
                if entity is EntityKind.USER
                else config.baseline.min_events_per_source
            ),
            value_range=value_range,
            null_semantics=absent,
            description=description,
            requires_baseline=True,
        )

    def coverage(name: str, description: str) -> FeatureSpec:
        return FeatureSpec(
            name=name,
            group=FeatureGroup.BASELINE,
            entity=EntityKind.NONE,
            aggregate=AggregateKind.FLAG,
            dtype=FeatureDType.BOOL,
            nullable=False,
            leakage_class=LeakageClass.BASELINE_DERIVED,
            null_semantics="Never null.",
            description=description,
            requires_baseline=True,
        )

    return (
        novelty(
            "is_new_device_for_user",
            "Whether the anchor's device is absent from this user's known devices.",
        ),
        novelty(
            "is_new_source_for_user",
            "Whether the anchor's source is absent from this user's known sources.",
        ),
        novelty(
            "is_new_country_for_user",
            "Whether the anchor's country is absent from this user's known countries.",
        ),
        novelty(
            "is_new_application_for_user",
            "Whether the anchor's application is absent from this user's known set.",
        ),
        novelty(
            "is_new_auth_method_for_user",
            "Whether the anchor's authentication method is absent from the known set.",
        ),
        numeric(
            "login_hour_deviation",
            EntityKind.USER,
            AggregateKind.DELTA,
            "unitless",
            (
                "How unusual the anchor's UTC hour is for this user, expressed as "
                "one minus the smoothed probability mass at that hour."
            ),
            value_range=(0.0, 1.0),
        ),
        numeric(
            "user_success_rate_deviation",
            EntityKind.USER,
            AggregateKind.DELTA,
            "ratio",
            (
                "Signed difference between the anchor outcome treated as a success "
                "indicator and this user's baseline success rate."
            ),
            value_range=(-1.0, 1.0),
        ),
        numeric(
            "user_event_rate_ratio",
            EntityKind.USER,
            AggregateKind.RATIO,
            "ratio",
            (
                "Recent observed event rate for this user divided by their "
                "baseline event rate."
            ),
        ),
        numeric(
            "response_time_zscore",
            EntityKind.USER,
            AggregateKind.ZSCORE,
            "zscore",
            (
                "Standardised anchor response time against this user's baseline "
                "response-time distribution, clipped to the configured maximum."
            ),
        ),
        numeric(
            "source_user_fanout_ratio",
            EntityKind.SOURCE,
            AggregateKind.RATIO,
            "ratio",
            (
                "Recent distinct-user count for this source divided by its "
                "baseline targeted-user count."
            ),
        ),
        numeric(
            "source_event_rate_ratio",
            EntityKind.SOURCE,
            AggregateKind.RATIO,
            "ratio",
            (
                "Recent observed event rate for this source divided by its "
                "baseline event rate."
            ),
        ),
        numeric(
            "distance_from_user_baseline_centroid_km",
            EntityKind.USER,
            AggregateKind.DELTA,
            "km",
            (
                "Great-circle distance from the anchor's coarse coordinates to "
                "this user's baseline location centroid."
            ),
            value_range=(0.0, 20100.0),
        ),
        coverage(
            "user_in_baseline",
            "Whether this user was present in the fitted baseline.",
        ),
        coverage(
            "source_in_baseline",
            "Whether this source was present in the fitted baseline.",
        ),
    )


class FeatureCatalog:
    """An immutable, ordered collection of :class:`FeatureSpec`.

    Column order is fixed at construction and is the authoritative ordering for
    every artifact this package writes.
    """

    __slots__ = ("_by_name", "_config_fingerprint", "_specs")

    def __init__(
        self, specs: tuple[FeatureSpec, ...], *, config_fingerprint: str
    ) -> None:
        seen: dict[str, FeatureSpec] = {}
        for spec in specs:
            if spec.name in seen:
                raise FeatureComputationError(
                    f"Duplicate feature name in catalog: {spec.name!r}"
                )
            seen[spec.name] = spec
        self._specs = specs
        self._by_name = seen
        self._config_fingerprint = config_fingerprint

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Any:
        return iter(self._specs)

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        """Return every spec in catalog order."""
        return self._specs

    @property
    def config_fingerprint(self) -> str:
        """Return the fingerprint of the config this catalog was built from."""
        return self._config_fingerprint

    def column_order(self) -> tuple[str, ...]:
        """Return the authoritative column ordering for all feature artifacts."""
        return tuple(spec.name for spec in self._specs)

    def get(self, name: str) -> FeatureSpec:
        """Return the spec named *name*.

        Raises:
            FeatureComputationError: if no such feature is declared.
        """
        try:
            return self._by_name[name]
        except KeyError:
            raise FeatureComputationError(
                f"Undeclared feature column: {name!r}"
            ) from None

    def has(self, name: str) -> bool:
        """Return whether *name* is a declared feature."""
        return name in self._by_name

    def specs_for_group(self, group: FeatureGroup) -> tuple[FeatureSpec, ...]:
        """Return every spec in *group*, in catalog order."""
        return tuple(s for s in self._specs if s.group is group)

    def specs_for_entity(self, entity: EntityKind) -> tuple[FeatureSpec, ...]:
        """Return every spec computed over *entity*, in catalog order."""
        return tuple(s for s in self._specs if s.entity is entity)

    def specs_for_leakage_class(
        self, leakage_class: LeakageClass
    ) -> tuple[FeatureSpec, ...]:
        """Return every spec with the given leakage classification.

        The leakage auditor uses this rather than matching column names.
        """
        return tuple(s for s in self._specs if s.leakage_class is leakage_class)

    def group_counts(self) -> dict[str, int]:
        """Return the number of features per group, for reporting."""
        counts: dict[str, int] = {}
        for spec in self._specs:
            counts[str(spec.group)] = counts.get(str(spec.group), 0) + 1
        return counts

    def empty_snapshot(self, entity: EntityKind) -> dict[str, Any]:
        """Return the values emitted for *entity* when it has no history.

        Covers the entity's *windowed* features only: nullable features are
        null, non-nullable count-like features are zero.  Sequence, geospatial,
        and baseline features are computed from separate state that always
        exists, so they are not part of an empty window snapshot.
        """
        snapshot: dict[str, Any] = {}
        for spec in self._specs:
            if spec.entity is not entity or spec.window is None:
                continue
            if spec.nullable:
                snapshot[spec.name] = None
            elif spec.is_count_like:
                snapshot[spec.name] = 0
            else:
                continue
        return snapshot

    def fingerprint(self) -> str:
        """Return a SHA-256 digest of the catalog's machine-readable contract.

        Covers names, groups, entities, windows, aggregates, types,
        nullability, units, leakage classes, thresholds, ranges, and privacy
        classification.  Excludes ``description`` and ``null_semantics`` so
        that correcting prose does not invalidate every artifact that records
        this fingerprint.
        """
        payload = {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "specs": [
                {
                    field: _json_scalar(getattr(spec, field))
                    for field in _FINGERPRINT_FIELDS
                }
                for spec in sorted(self._specs, key=lambda s: s.name)
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def _json_scalar(value: object) -> Any:
    """Render a spec field into a JSON-stable scalar."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, tuple):
        return [_json_scalar(v) for v in value]
    if isinstance(value, float):
        return "inf" if value == float("inf") else f"{value:.9f}"
    return str(value)


def _reject_unsafe_names(specs: tuple[FeatureSpec, ...]) -> None:
    """Reject prohibited or conclusion-asserting feature names."""
    names = {spec.name for spec in specs}

    prohibited = sorted(names & PROHIBITED_FEATURE_COLUMNS)
    if prohibited:
        raise FeatureComputationError(
            f"Catalog declares prohibited column(s): {prohibited}"
        )

    for name in sorted(names):
        tokens = set(name.split("__")[0].split("_"))
        overlap = tokens & AMBIGUOUS_FEATURE_NAMES
        if overlap or name in AMBIGUOUS_FEATURE_NAMES:
            raise FeatureComputationError(
                f"Feature name {name!r} uses ambiguous or conclusion-asserting "
                f"term(s): {sorted(overlap) or [name]}"
            )


def build_catalog(config: FeatureConfig) -> FeatureCatalog:
    """Build the feature catalog implied by *config*.

    Windowed features are expanded over :meth:`FeatureConfig.windows_for`, so
    the width trims declared in configuration propagate to the schema, the
    Arrow types, the writer, and the validator together.
    """
    specs: list[FeatureSpec] = []
    specs.extend(_key_specs())
    specs.extend(_current_context_specs())
    specs.extend(_calendar_specs())

    for kind, templates in _TEMPLATE_MATRIX:
        for template in templates:
            for window in config.windows_for(kind, template.aggregate):
                specs.append(_expand(template, kind, window, config))

    specs.extend(_sequence_specs())
    if config.geospatial.enabled:
        specs.extend(_geospatial_specs())
    if config.baseline.enabled:
        specs.extend(_baseline_specs(config))

    frozen = tuple(specs)
    _reject_unsafe_names(frozen)
    return FeatureCatalog(frozen, config_fingerprint=config.fingerprint())


def catalog_to_markdown(catalog: FeatureCatalog) -> str:
    """Render the catalog as Markdown documentation.

    Contains only schema metadata: no data, no identifiers, no paths.
    """
    lines: list[str] = [
        "# Feature Catalog",
        "",
        "Generated from the feature catalog. Do not edit by hand; regenerate "
        "with `password-attack-detector features catalog --format markdown`.",
        "",
        f"- Feature schema version: `{FEATURE_SCHEMA_VERSION}`",
        f"- Catalog fingerprint: `{catalog.fingerprint()}`",
        f"- Configuration fingerprint: `{catalog.config_fingerprint}`",
        f"- Declared features: {len(catalog)}",
        "",
        "## Feature counts by group",
        "",
        "| Group | Features |",
        "|--------|-------|",
    ]
    for group, count in sorted(catalog.group_counts().items()):
        lines.append(f"| `{group}` | {count} |")

    lines.extend(
        [
            "",
            "## Leakage classification",
            "",
            "| Class | Meaning | Features |",
            "|--------|-------|-------|",
            "| `key` | Identity and timing; never a model input | "
            f"{len(catalog.specs_for_leakage_class(LeakageClass.KEY))} |",
            "| `current_event_context` | The anchor event's own recorded fields, "
            "legitimate because detection runs after the event completed | "
            f"{len(catalog.specs_for_leakage_class(LeakageClass.CURRENT_EVENT_CONTEXT))} |",
            "| `prior_only` | Computed strictly from events in `[t - window, t)` | "
            f"{len(catalog.specs_for_leakage_class(LeakageClass.PRIOR_ONLY))} |",
            "| `baseline_derived` | Derived from a baseline fitted on an approved "
            "reference interval | "
            f"{len(catalog.specs_for_leakage_class(LeakageClass.BASELINE_DERIVED))} |",
        ]
    )

    for group in FeatureGroup:
        group_specs = catalog.specs_for_group(group)
        if not group_specs:
            continue
        lines.extend(
            [
                "",
                f"## {str(group).replace('_', ' ').title()}",
                "",
                "| Feature | Type | Nullable | Unit | Window | Leakage class | "
                "Null semantics | Description |",
                "|--------|-------|-------|-------|-------|-------|-------|-------|",
            ]
        )
        for spec in group_specs:
            lines.append(
                f"| `{spec.name}` | `{spec.dtype}` | "
                f"{'yes' if spec.nullable else 'no'} | {spec.unit or '-'} | "
                f"{spec.window or '-'} | `{spec.leakage_class}` | "
                f"{spec.null_semantics} | {spec.description} |"
            )

    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- Feature names describe observations and deviations, never "
            "conclusions. No column asserts that an attack occurred.",
            "- Ground-truth labels, campaign metadata, and split assignments are "
            "absent by construction and are rejected at catalog build time.",
            "- Baseline-derived features are null for entities absent from the "
            "fitted baseline; they are not silently treated as novel.",
            "- Synthetic data exercises these features but does not demonstrate "
            "real-world detection effectiveness.",
            "",
        ]
    )
    return "\n".join(lines)
