"""Tests for the pure geospatial calculations."""

from __future__ import annotations

import math

import pytest

from password_attack_detector.features.geospatial import (
    EARTH_RADIUS_KM,
    MAX_EARTH_DISTANCE_KM,
    GeoStatus,
    VelocityStatus,
    haversine_km,
    implied_velocity_kmh,
    previous_success_geo,
    valid_coordinates,
)

# Reference coordinates and great-circle distances, in kilometres.  Expected
# values come from published great-circle figures; the tolerance reflects the
# spread between common Earth-radius conventions, not implementation slop.
_LONDON = (51.5074, -0.1278)
_PARIS = (48.8566, 2.3522)
_NEW_YORK = (40.7128, -74.0060)
_SYDNEY = (-33.8688, 151.2093)
_SAN_FRANCISCO = (37.7749, -122.4194)


class TestCoordinateValidation:
    @pytest.mark.parametrize(
        "latitude,longitude",
        [(0.0, 0.0), (90.0, 180.0), (-90.0, -180.0), (51.5, -0.1), (-33.9, 151.2)],
    )
    def test_valid_pairs_accepted(self, latitude: float, longitude: float) -> None:
        assert valid_coordinates(latitude, longitude)

    @pytest.mark.parametrize(
        "latitude,longitude",
        [
            (None, 0.0),
            (0.0, None),
            (None, None),
            (90.001, 0.0),
            (-90.001, 0.0),
            (0.0, 180.001),
            (0.0, -180.001),
        ],
    )
    def test_invalid_pairs_rejected(
        self, latitude: float | None, longitude: float | None
    ) -> None:
        assert not valid_coordinates(latitude, longitude)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_values_rejected(self, value: float) -> None:
        assert not valid_coordinates(value, 0.0)
        assert not valid_coordinates(0.0, value)


class TestHaversine:
    @pytest.mark.parametrize(
        "a,b,expected_km",
        [
            (_LONDON, _PARIS, 343.0),
            (_LONDON, _NEW_YORK, 5570.0),
            (_NEW_YORK, _SAN_FRANCISCO, 4129.0),
            (_LONDON, _SYDNEY, 16993.0),
        ],
    )
    def test_known_distances(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        expected_km: float,
    ) -> None:
        assert haversine_km(a[0], a[1], b[0], b[1]) == pytest.approx(
            expected_km, rel=0.005
        )

    def test_identical_points_are_zero(self) -> None:
        assert haversine_km(51.5, -0.1, 51.5, -0.1) == 0.0

    def test_distance_is_symmetric(self) -> None:
        forward = haversine_km(*_LONDON, *_SYDNEY)
        backward = haversine_km(*_SYDNEY, *_LONDON)
        assert forward == pytest.approx(backward)

    def test_antipodal_points_are_half_the_circumference(self) -> None:
        assert haversine_km(0.0, 0.0, 0.0, 180.0) == pytest.approx(
            MAX_EARTH_DISTANCE_KM
        )
        assert haversine_km(45.0, 10.0, -45.0, -170.0) == pytest.approx(
            MAX_EARTH_DISTANCE_KM
        )

    def test_pole_to_pole(self) -> None:
        assert haversine_km(90.0, 0.0, -90.0, 0.0) == pytest.approx(
            MAX_EARTH_DISTANCE_KM
        )

    def test_near_antipodal_points_do_not_raise(self) -> None:
        # Rounding can push the haversine term a few ulps above 1.0; clamping
        # is what stops asin from raising a domain error here.
        for offset in (1e-12, 1e-9, 1e-6):
            distance = haversine_km(0.0, 0.0, -offset, 180.0 - offset)
            assert math.isfinite(distance)
            assert distance <= MAX_EARTH_DISTANCE_KM + 1e-6

    def test_no_result_exceeds_half_the_circumference(self) -> None:
        for lat_a in range(-90, 91, 30):
            for lon_a in range(-180, 181, 60):
                for lat_b in range(-90, 91, 30):
                    for lon_b in range(-180, 181, 60):
                        distance = haversine_km(
                            float(lat_a), float(lon_a), float(lat_b), float(lon_b)
                        )
                        assert 0.0 <= distance <= MAX_EARTH_DISTANCE_KM + 1e-6

    def test_one_degree_of_latitude_is_about_111_km(self) -> None:
        assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.2, rel=0.01)

    def test_longitude_degrees_shrink_towards_the_poles(self) -> None:
        equator = haversine_km(0.0, 0.0, 0.0, 1.0)
        high = haversine_km(60.0, 0.0, 60.0, 1.0)
        assert high == pytest.approx(equator * 0.5, rel=0.01)

    def test_earth_radius_constant_is_the_iugg_mean(self) -> None:
        assert pytest.approx(6371.0088) == EARTH_RADIUS_KM

    @pytest.mark.parametrize(
        "args",
        [
            (91.0, 0.0, 0.0, 0.0),
            (0.0, 181.0, 0.0, 0.0),
            (0.0, 0.0, -91.0, 0.0),
            (float("nan"), 0.0, 0.0, 0.0),
        ],
    )
    def test_invalid_input_raises(self, args: tuple[float, ...]) -> None:
        with pytest.raises(ValueError, match="valid, finite coordinates"):
            haversine_km(*args)


class TestImpliedVelocity:
    def test_computes_distance_over_time(self) -> None:
        status, velocity = implied_velocity_kmh(100.0, 3600.0, max_plausible_kmh=1200.0)
        assert status is VelocityStatus.OK
        assert velocity == pytest.approx(100.0)

    def test_zero_elapsed_is_reported_not_divided(self) -> None:
        status, velocity = implied_velocity_kmh(100.0, 0.0, max_plausible_kmh=1200.0)
        assert status is VelocityStatus.ZERO_ELAPSED
        assert velocity is None

    def test_negative_elapsed_is_also_rejected(self) -> None:
        status, velocity = implied_velocity_kmh(100.0, -5.0, max_plausible_kmh=1200.0)
        assert status is VelocityStatus.ZERO_ELAPSED
        assert velocity is None

    def test_extreme_values_are_capped_and_stay_finite(self) -> None:
        status, velocity = implied_velocity_kmh(20000.0, 1.0, max_plausible_kmh=1200.0)
        assert status is VelocityStatus.CAPPED
        assert velocity == pytest.approx(1200.0)
        assert velocity is not None and math.isfinite(velocity)

    def test_value_exactly_at_the_cap_is_not_flagged(self) -> None:
        status, velocity = implied_velocity_kmh(
            1200.0, 3600.0, max_plausible_kmh=1200.0
        )
        assert status is VelocityStatus.OK
        assert velocity == pytest.approx(1200.0)

    def test_sub_threshold_distance_is_treated_as_stationary(self) -> None:
        status, velocity = implied_velocity_kmh(
            0.5, 1.0, max_plausible_kmh=1200.0, min_distance_km=5.0
        )
        assert status is VelocityStatus.OK
        assert velocity == 0.0

    def test_zero_distance_gives_zero_velocity(self) -> None:
        status, velocity = implied_velocity_kmh(0.0, 3600.0, max_plausible_kmh=1200.0)
        assert status is VelocityStatus.OK
        assert velocity == pytest.approx(0.0)


class TestPreviousSuccessGeo:
    def test_no_prior_success(self) -> None:
        result = previous_success_geo(
            current_latitude=51.5,
            current_longitude=-0.1,
            prior_latitude=None,
            prior_longitude=None,
            prior_elapsed_seconds=None,
            has_prior_success=False,
            max_plausible_kmh=1200.0,
        )
        assert result.status is GeoStatus.NO_PRIOR_SUCCESS
        assert result.distance_km is None
        assert result.velocity_status is VelocityStatus.UNAVAILABLE

    def test_missing_current_location(self) -> None:
        result = previous_success_geo(
            current_latitude=None,
            current_longitude=None,
            prior_latitude=51.5,
            prior_longitude=-0.1,
            prior_elapsed_seconds=60.0,
            has_prior_success=True,
            max_plausible_kmh=1200.0,
        )
        assert result.status is GeoStatus.MISSING_CURRENT_LOCATION
        assert result.distance_km is None

    def test_missing_prior_location(self) -> None:
        result = previous_success_geo(
            current_latitude=51.5,
            current_longitude=-0.1,
            prior_latitude=None,
            prior_longitude=None,
            prior_elapsed_seconds=None,
            has_prior_success=True,
            max_plausible_kmh=1200.0,
        )
        assert result.status is GeoStatus.MISSING_PRIOR_LOCATION
        assert result.distance_km is None

    def test_full_result_when_everything_is_available(self) -> None:
        result = previous_success_geo(
            current_latitude=_PARIS[0],
            current_longitude=_PARIS[1],
            prior_latitude=_LONDON[0],
            prior_longitude=_LONDON[1],
            prior_elapsed_seconds=3600.0,
            has_prior_success=True,
            max_plausible_kmh=1200.0,
        )
        assert result.status is GeoStatus.OK
        assert result.distance_km == pytest.approx(343.0, rel=0.005)
        assert result.velocity_status is VelocityStatus.OK
        assert result.velocity_kmh == pytest.approx(343.0, rel=0.005)

    def test_simultaneous_prior_success_yields_zero_elapsed(self) -> None:
        result = previous_success_geo(
            current_latitude=_PARIS[0],
            current_longitude=_PARIS[1],
            prior_latitude=_LONDON[0],
            prior_longitude=_LONDON[1],
            prior_elapsed_seconds=0.0,
            has_prior_success=True,
            max_plausible_kmh=1200.0,
        )
        assert result.status is GeoStatus.OK
        assert result.distance_km is not None
        assert result.velocity_status is VelocityStatus.ZERO_ELAPSED
        assert result.velocity_kmh is None

    def test_result_is_immutable(self) -> None:
        result = previous_success_geo(
            current_latitude=None,
            current_longitude=None,
            prior_latitude=None,
            prior_longitude=None,
            prior_elapsed_seconds=None,
            has_prior_success=False,
            max_plausible_kmh=1200.0,
        )
        with pytest.raises(AttributeError):
            result.distance_km = 5.0  # type: ignore[misc]


class TestNoAttackDecision:
    def test_module_exposes_no_travel_verdict(self) -> None:
        from password_attack_detector.features import geospatial

        exported = set(geospatial.__all__)
        for banned in ("impossible_travel", "is_impossible", "travel_verdict"):
            assert banned not in exported

    def test_status_values_describe_availability_not_conclusions(self) -> None:
        names: list[str] = [str(s) for s in GeoStatus]
        names.extend(str(s) for s in VelocityStatus)
        for name in names:
            assert "impossible" not in name
            assert "attack" not in name
