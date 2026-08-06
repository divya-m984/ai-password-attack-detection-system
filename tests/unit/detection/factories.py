"""Deterministic feature-snapshot builders for the detection tests.

Every builder starts from a **complete, catalog-shaped row**: one entry for
each column the feature catalog declares, at a value that means "nothing
happened here".  A test then overrides only the handful of columns it is
actually about.

That inversion is the point.  A test that hand-listed the columns one rule
reads would break the moment the catalog grew a column, and -- worse -- a rule
that quietly started reading a new column would still pass.  Starting from the
full shape means a snapshot is always schema-complete, and a test's overrides
read as a statement about the scenario rather than about the schema.

Values are fixed literals.  No random state, no wall clock, no identifier that
varies between runs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Final

from password_attack_detector.data.enums import AuthOutcome, MFAOutcome
from password_attack_detector.features.catalog import (
    ANCHOR_EVENT_ID,
    ANCHOR_EVENT_TIME,
    FeatureCatalog,
    FeatureDType,
    build_catalog,
)
from password_attack_detector.features.config import (
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
)

#: A fixed anchor identifier.  Shaped like the event identifiers Phase 2 emits
#: but constant, so a detection identifier derived from it is reproducible.
ANCHOR: Final[str] = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"

#: A fixed anchor timestamp.  Timezone-aware, as the schema requires.
WHEN: Final[datetime] = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)

#: Non-nullable string columns take a benign default rather than the empty
#: string, because a rule comparing against a vocabulary needs a member of it.
_STRING_DEFAULTS: Final[dict[str, str]] = {
    "feature_schema_version": FEATURE_SCHEMA_VERSION,
    "current_authentication_outcome": str(AuthOutcome.FAILURE),
    "current_authentication_method": "password",
    # The statuses that mean "this comparison could not be made", which is the
    # correct baseline for a row where nothing has happened yet.
    "user_previous_success_geo__status": "no_prior_success",
    "implied_velocity__status": "unavailable",
}


def feature_catalog() -> FeatureCatalog:
    """Return the catalog implied by the default feature configuration."""
    return build_catalog(FeatureConfig())


def _default_for(name: str, dtype: FeatureDType, nullable: bool) -> Any:
    """Return the quiet-baseline value for one column."""
    if name == ANCHOR_EVENT_ID:
        return ANCHOR
    if name == ANCHOR_EVENT_TIME:
        return WHEN
    if nullable:
        # A nullable column's null means "not observed", which is exactly the
        # baseline a scenario-free row should carry.
        return None
    match dtype:
        case FeatureDType.INT64:
            return 0
        case FeatureDType.FLOAT64:
            return 0.0
        case FeatureDType.BOOL:
            return False
        case FeatureDType.TIMESTAMP:
            return WHEN
        case FeatureDType.STRING:
            return _STRING_DEFAULTS.get(name, "unknown")


def snapshot(
    catalog: FeatureCatalog | None = None,
    *,
    anchor_event_id: str = ANCHOR,
    anchor_event_time: datetime = WHEN,
    **overrides: Any,
) -> dict[str, Any]:
    """Return a complete feature-snapshot row with *overrides* applied.

    Raises:
        KeyError: if an override names a column the catalog does not declare.
            A typo in a column name would otherwise produce a test that passes
            because the rule read the untouched default.
    """
    resolved = feature_catalog() if catalog is None else catalog
    row: dict[str, Any] = {
        spec.name: _default_for(spec.name, spec.dtype, spec.nullable)
        for spec in resolved.specs
    }
    row[ANCHOR_EVENT_ID] = anchor_event_id
    row[ANCHOR_EVENT_TIME] = anchor_event_time

    unknown = sorted(name for name in overrides if name not in row)
    if unknown:
        raise KeyError(f"snapshot override names undeclared column(s) {unknown}")
    row.update(overrides)
    return row


def quiet_row(
    catalog: FeatureCatalog | None = None, **overrides: Any
) -> dict[str, Any]:
    """A snapshot with history observed and nothing suspicious in it.

    Distinct from a bare :func:`snapshot`, where every nullable column is null.
    A null rate means *unobserved*, and a rule that sees one correctly reports
    ``insufficient_data``.  This row populates the minimum-history columns with
    benign values, so a rule evaluating it reaches an actual verdict -- which
    is what a test asserting ``not_fired`` needs.
    """
    return snapshot(
        catalog,
        **{
            "pair_failure_rate__5m": 0.0,
            "source_failure_rate__1h": 0.0,
            "user_failure_rate__1h": 0.0,
            "source_interarrival_coefficient_of_variation__15m": 2.5,
            "source_mean_interarrival_seconds__15m": 600.0,
            "user_in_baseline": True,
            "is_new_device_for_user": False,
            "is_new_source_for_user": False,
            "is_new_country_for_user": False,
            "is_new_application_for_user": False,
            "is_new_auth_method_for_user": False,
            "current_mfa_outcome": str(MFAOutcome.PASSED),
            "user_attempt_count__15m": 10,
            "user_challenge_count__15m": 5,
            **overrides,
        },
    )


# ---------------------------------------------------------------------------
# Scenario rows: the smallest override set that fires each rule at its defaults
#
# Each returns a firing snapshot.  A test that wants a negative case overrides
# the one condition it is testing, so the failure is unambiguous.
# ---------------------------------------------------------------------------


def brute_force_row(
    catalog: FeatureCatalog | None = None, **overrides: Any
) -> dict[str, Any]:
    """A snapshot that fires PAD-BF-001 at its declared defaults."""
    return snapshot(
        catalog,
        **{
            "pair_failure_count__5m": 12,
            "pair_failure_rate__5m": 0.95,
            "user_failure_count__5m": 14,
            "prior_consecutive_user_failures": 9,
            "source_unique_user_count__5m": 1,
            "pair_mean_interarrival_seconds__5m": 4.0,
            "user_blocked_count__5m": 3,
            **overrides,
        },
    )


def success_after_burst_row(
    catalog: FeatureCatalog | None = None, **overrides: Any
) -> dict[str, Any]:
    """A snapshot that fires PAD-BF-002 at its declared defaults."""
    return snapshot(
        catalog,
        **{
            "current_authentication_outcome": str(AuthOutcome.SUCCESS),
            "prior_failures_since_pair_success": 9,
            "prior_failures_since_user_success": 11,
            "previous_user_outcome": str(AuthOutcome.FAILURE),
            "seconds_since_user_previous_failure": 20.0,
            **overrides,
        },
    )


def spraying_row(
    catalog: FeatureCatalog | None = None, **overrides: Any
) -> dict[str, Any]:
    """A snapshot that fires PAD-PS-001 at its declared defaults."""
    return snapshot(
        catalog,
        **{
            "source_unique_user_count__1h": 40,
            "source_failure_count__1h": 76,
            "source_failure_rate__1h": 0.95,
            "source_attempt_count__1h": 80,
            **overrides,
        },
    )


def stuffing_row(
    catalog: FeatureCatalog | None = None, **overrides: Any
) -> dict[str, Any]:
    """A snapshot that fires PAD-CS-001 at its declared defaults."""
    return snapshot(
        catalog,
        **{
            "source_unique_user_count__1h": 30,
            "source_success_count__1h": 4,
            "source_failure_count__1h": 56,
            "source_attempt_count__1h": 60,
            "source_unique_device_count__1h": 9,
            "source_unique_user_agent_count__1h": 7,
            "user_in_baseline": True,
            "is_new_device_for_user": True,
            "is_new_country_for_user": True,
            **overrides,
        },
    )


def distributed_row(
    catalog: FeatureCatalog | None = None, **overrides: Any
) -> dict[str, Any]:
    """A snapshot that fires PAD-DBF-001 at its declared defaults."""
    return snapshot(
        catalog,
        **{
            "user_unique_source_count__1h": 24,
            "user_failure_count__1h": 60,
            "user_failure_rate__1h": 0.96,
            "pair_attempt_count__1h": 2,
            "source_unique_user_count__1h": 1,
            **overrides,
        },
    )


def takeover_row(
    catalog: FeatureCatalog | None = None, **overrides: Any
) -> dict[str, Any]:
    """A snapshot that fires PAD-ATO-001 at its declared defaults."""
    return snapshot(
        catalog,
        **{
            "current_authentication_outcome": str(AuthOutcome.SUCCESS),
            "user_in_baseline": True,
            "is_new_device_for_user": True,
            "is_new_country_for_user": True,
            "is_new_source_for_user": False,
            "is_new_application_for_user": False,
            "is_new_auth_method_for_user": False,
            "prior_failures_since_user_success": 7,
            **overrides,
        },
    )


def impossible_travel_row(
    catalog: FeatureCatalog | None = None, **overrides: Any
) -> dict[str, Any]:
    """A snapshot that fires PAD-GEO-001 at its declared defaults."""
    return snapshot(
        catalog,
        **{
            "current_authentication_outcome": str(AuthOutcome.SUCCESS),
            "user_previous_success_geo__status": "ok",
            "implied_velocity__status": "ok",
            "distance_km_from_user_previous_success": 4200.0,
            "implied_velocity_kmh_from_previous_success": 3600.0,
            "country_changed_since_previous_success": True,
            "seconds_since_user_previous_success_with_location": 4200.0,
            **overrides,
        },
    )


def bot_row(catalog: FeatureCatalog | None = None, **overrides: Any) -> dict[str, Any]:
    """A snapshot that fires PAD-BOT-001 at its declared defaults."""
    return snapshot(
        catalog,
        **{
            "source_attempt_count__15m": 90,
            "source_interarrival_coefficient_of_variation__15m": 0.02,
            "source_mean_interarrival_seconds__15m": 10.0,
            "source_unique_user_agent_count__1h": 1,
            "source_unique_user_count__1h": 5,
            **overrides,
        },
    )


def mfa_row(catalog: FeatureCatalog | None = None, **overrides: Any) -> dict[str, Any]:
    """A snapshot that fires PAD-MFA-001 at its declared defaults."""
    return snapshot(
        catalog,
        **{
            "user_attempt_count__15m": 20,
            "user_mfa_failure_count__15m": 9,
            "user_challenge_count__15m": 10,
            "current_mfa_outcome": str(MFAOutcome.FAILED),
            **overrides,
        },
    )
