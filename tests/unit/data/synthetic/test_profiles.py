"""Unit tests for password_attack_detector.data.synthetic.profiles."""

from __future__ import annotations

from password_attack_detector.data.enums import AuthMethod, ClientType
from password_attack_detector.data.synthetic.profiles import (
    ApplicationProfile,
    DeviceProfile,
    SourceProfile,
    UserProfile,
)


class TestUserProfile:
    def test_construction(self) -> None:
        up = UserProfile(
            user_id="u:" + "a" * 32,
            normal_login_hours=(8, 9, 10, 11, 12, 13, 14, 15),
            home_countries=("US",),
            known_device_ids=("d:" + "b" * 32,),
            known_source_ids=("s:" + "c" * 32,),
            preferred_auth_methods=(AuthMethod.PASSWORD,),
            baseline_success_rate=0.95,
        )
        assert up.user_id == "u:" + "a" * 32
        assert up.baseline_success_rate == 0.95

    def test_is_frozen(self) -> None:
        up = UserProfile(
            user_id="u:" + "a" * 32,
            normal_login_hours=(9, 10),
            home_countries=("US",),
            known_device_ids=("d:" + "b" * 32,),
            known_source_ids=("s:" + "c" * 32,),
            preferred_auth_methods=(AuthMethod.PASSWORD,),
            baseline_success_rate=0.9,
        )
        import dataclasses

        assert dataclasses.is_dataclass(up)
        # frozen means no assignment (FrozenInstanceError)
        try:
            up.baseline_success_rate = 0.5  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except Exception as exc:
            msg = str(exc).lower()
            assert "frozen" in msg or "can't" in msg or "cannot" in msg

    def test_normal_login_hours_is_tuple(self) -> None:
        up = UserProfile(
            user_id="u:" + "a" * 32,
            normal_login_hours=(8, 9, 10),
            home_countries=("DE",),
            known_device_ids=(),
            known_source_ids=(),
            preferred_auth_methods=(AuthMethod.PASSWORD,),
            baseline_success_rate=0.9,
        )
        assert isinstance(up.normal_login_hours, tuple)

    def test_multiple_auth_methods(self) -> None:
        up = UserProfile(
            user_id="u:" + "a" * 32,
            normal_login_hours=(9, 10),
            home_countries=("US",),
            known_device_ids=(),
            known_source_ids=(),
            preferred_auth_methods=(AuthMethod.PASSWORD, AuthMethod.MFA_TOTP),
            baseline_success_rate=0.95,
        )
        assert len(up.preferred_auth_methods) == 2


class TestSourceProfile:
    def test_construction(self) -> None:
        sp = SourceProfile(
            source_id="s:" + "a" * 32,
            country_code="US",
            coarse_latitude=38.0,
            coarse_longitude=-97.0,
        )
        assert sp.country_code == "US"
        assert sp.coarse_latitude == 38.0
        assert sp.coarse_longitude == -97.0

    def test_is_frozen(self) -> None:
        sp = SourceProfile(
            source_id="s:" + "a" * 32,
            country_code="US",
            coarse_latitude=38.0,
            coarse_longitude=-97.0,
        )
        try:
            sp.country_code = "DE"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except Exception as exc:
            msg = str(exc).lower()
            assert "frozen" in msg or "can't" in msg or "cannot" in msg


class TestDeviceProfile:
    def test_construction(self) -> None:
        dp = DeviceProfile(
            device_id="d:" + "a" * 32,
            user_agent_family="Chrome",
            operating_system_family="Windows",
            client_type=ClientType.WEB_BROWSER,
        )
        assert dp.user_agent_family == "Chrome"
        assert dp.client_type == ClientType.WEB_BROWSER

    def test_client_type_is_enum(self) -> None:
        dp = DeviceProfile(
            device_id="d:" + "a" * 32,
            user_agent_family="curl",
            operating_system_family="Linux",
            client_type=ClientType.CLI_TOOL,
        )
        assert isinstance(dp.client_type, ClientType)


class TestApplicationProfile:
    def test_construction(self) -> None:
        ap = ApplicationProfile(application_id="app-00")
        assert ap.application_id == "app-00"

    def test_is_frozen(self) -> None:
        ap = ApplicationProfile(application_id="app-01")
        try:
            ap.application_id = "app-99"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except Exception as exc:
            msg = str(exc).lower()
            assert "frozen" in msg or "can't" in msg or "cannot" in msg
