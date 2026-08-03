"""Pure geospatial calculations over coarse coordinates.

Everything here is a total function of its arguments: no state, no I/O, no
configuration lookups beyond the values passed in.

This module deliberately produces **no attack decision**.  There is no
``impossible_travel`` flag and no travel-feasibility verdict.  It reports a
distance, an elapsed interval, and a normalised implied velocity, together
with a status explaining why a value is or is not available.  Interpreting
those magnitudes is a later phase's responsibility.

Coordinates in the canonical event schema are deliberately coarse, so derived
distances carry correspondingly coarse precision.  Two events in the same
metropolitan area may report a non-zero distance purely from rounding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "EARTH_RADIUS_KM",
    "MAX_EARTH_DISTANCE_KM",
    "GeoStatus",
    "PreviousSuccessGeo",
    "VelocityStatus",
    "haversine_km",
    "implied_velocity_kmh",
    "valid_coordinates",
]

#: IUGG mean Earth radius, in kilometres.
EARTH_RADIUS_KM: float = 6371.0088

#: Half the Earth's circumference: the greatest possible great-circle distance.
MAX_EARTH_DISTANCE_KM: float = math.pi * EARTH_RADIUS_KM


class GeoStatus(StrEnum):
    """Why a previous-success distance is or is not available."""

    OK = "ok"
    NO_PRIOR_SUCCESS = "no_prior_success"
    MISSING_CURRENT_LOCATION = "missing_current_location"
    MISSING_PRIOR_LOCATION = "missing_prior_location"


class VelocityStatus(StrEnum):
    """Why an implied velocity is or is not available."""

    OK = "ok"
    #: The two events share a timestamp, so a rate is undefined.
    ZERO_ELAPSED = "zero_elapsed"
    #: A finite value was computed but exceeded the configured plausible cap.
    CAPPED = "capped"
    #: No distance was available, so no velocity could be derived.
    UNAVAILABLE = "unavailable"


def valid_coordinates(latitude: float | None, longitude: float | None) -> bool:
    """Return whether the pair is a usable coordinate.

    Rejects ``None``, NaN, infinities, and out-of-range values.  Callers must
    check this before calling :func:`haversine_km`, which assumes valid input.
    """
    if latitude is None or longitude is None:
        return False
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        return False
    return -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0


def haversine_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return the great-circle distance between two points, in kilometres.

    Uses the haversine formula::

        a = sin^2(dlat / 2) + cos(lat_a) * cos(lat_b) * sin^2(dlon / 2)
        d = 2 * R * asin(sqrt(a))

    The ``asin(sqrt(a))`` form is used rather than ``atan2`` because it is
    well conditioned for the small distances that dominate real data.  ``a`` is
    clamped to ``[0, 1]`` before the square root: it is mathematically bounded
    by 1, but floating-point rounding can push antipodal pairs a few ulps past
    it, which would make ``asin`` raise.

    Raises:
        ValueError: if either coordinate pair is out of range or non-finite.
    """
    if not valid_coordinates(latitude_a, longitude_a) or not valid_coordinates(
        latitude_b, longitude_b
    ):
        raise ValueError("haversine_km requires valid, finite coordinates")

    phi_a = math.radians(latitude_a)
    phi_b = math.radians(latitude_b)
    delta_phi = phi_b - phi_a
    delta_lambda = math.radians(longitude_b - longitude_a)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2.0) ** 2
    )
    a = min(1.0, max(0.0, a))
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass(frozen=True, slots=True)
class PreviousSuccessGeo:
    """Distance, elapsed time, and velocity relative to a previous success."""

    status: GeoStatus
    distance_km: float | None
    elapsed_seconds: float | None
    velocity_status: VelocityStatus
    velocity_kmh: float | None


def implied_velocity_kmh(
    distance_km: float,
    elapsed_seconds: float,
    *,
    max_plausible_kmh: float,
    min_distance_km: float = 0.0,
) -> tuple[VelocityStatus, float | None]:
    """Return the implied velocity and the status explaining it.

    A zero elapsed interval yields ``ZERO_ELAPSED`` and a null value rather
    than a division by zero or an infinity.  Values above *max_plausible_kmh*
    are clipped to that ceiling and reported as ``CAPPED``, so the column stays
    finite and comparable without asserting that any particular movement was
    impossible.

    Distances below *min_distance_km* are treated as noise from coarse
    coordinates and reported as a velocity of zero.
    """
    if elapsed_seconds <= 0.0:
        return VelocityStatus.ZERO_ELAPSED, None
    if distance_km < min_distance_km:
        return VelocityStatus.OK, 0.0

    velocity = distance_km / (elapsed_seconds / 3600.0)
    if velocity > max_plausible_kmh:
        return VelocityStatus.CAPPED, max_plausible_kmh
    return VelocityStatus.OK, velocity


def previous_success_geo(
    *,
    current_latitude: float | None,
    current_longitude: float | None,
    prior_latitude: float | None,
    prior_longitude: float | None,
    prior_elapsed_seconds: float | None,
    has_prior_success: bool,
    max_plausible_kmh: float,
    min_distance_km: float = 0.0,
) -> PreviousSuccessGeo:
    """Compute the previous-success geospatial family in one pass.

    The status columns exist because null here is genuinely ambiguous: it can
    mean no prior success, no coordinates now, or no coordinates then.  Callers
    that need to distinguish those cases read the status rather than guessing.
    """
    if not has_prior_success:
        return PreviousSuccessGeo(
            status=GeoStatus.NO_PRIOR_SUCCESS,
            distance_km=None,
            elapsed_seconds=None,
            velocity_status=VelocityStatus.UNAVAILABLE,
            velocity_kmh=None,
        )

    if not valid_coordinates(current_latitude, current_longitude):
        return PreviousSuccessGeo(
            status=GeoStatus.MISSING_CURRENT_LOCATION,
            distance_km=None,
            elapsed_seconds=prior_elapsed_seconds,
            velocity_status=VelocityStatus.UNAVAILABLE,
            velocity_kmh=None,
        )

    if (
        not valid_coordinates(prior_latitude, prior_longitude)
        or prior_elapsed_seconds is None
    ):
        return PreviousSuccessGeo(
            status=GeoStatus.MISSING_PRIOR_LOCATION,
            distance_km=None,
            elapsed_seconds=prior_elapsed_seconds,
            velocity_status=VelocityStatus.UNAVAILABLE,
            velocity_kmh=None,
        )

    # Narrowing for the type checker: validity was established above.
    assert current_latitude is not None and current_longitude is not None
    assert prior_latitude is not None and prior_longitude is not None

    distance = haversine_km(
        prior_latitude, prior_longitude, current_latitude, current_longitude
    )
    velocity_status, velocity = implied_velocity_kmh(
        distance,
        prior_elapsed_seconds,
        max_plausible_kmh=max_plausible_kmh,
        min_distance_km=min_distance_km,
    )
    return PreviousSuccessGeo(
        status=GeoStatus.OK,
        distance_km=distance,
        elapsed_seconds=prior_elapsed_seconds,
        velocity_status=velocity_status,
        velocity_kmh=velocity,
    )
