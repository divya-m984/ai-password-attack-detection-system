"""Behavioral profile dataclasses for synthetic entity population.

Profiles describe the *behavioral* attributes of synthetic entities:
login-hour distributions, home geographies, device ownership, preferred
authentication methods, and baseline success rates.  All IDs within profiles
are deterministic pseudonyms produced by UUIDv5 (see ``entities.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from password_attack_detector.data.enums import AuthMethod, ClientType

__all__ = [
    "ApplicationProfile",
    "DeviceProfile",
    "SourceProfile",
    "UserProfile",
]


@dataclass(frozen=True, slots=True)
class UserProfile:
    """Behavioral descriptor for a synthetic user.

    Attributes
    ----------
    user_id:
        Prefixed pseudonym ``u:<hex32>`` generated via UUIDv5.
    normal_login_hours:
        Hours of the day (0-23) considered normal for this user.
    home_countries:
        ISO 3166-1 alpha-2 country codes the user typically authenticates from.
    known_device_ids:
        Device pseudonyms owned or regularly used by this user.
    known_source_ids:
        Source pseudonyms (network origins) typical for this user.
    preferred_auth_methods:
        Ordered list of authentication methods this user typically uses.
    baseline_success_rate:
        Expected fraction of this user's legitimate attempts that succeed.
    """

    user_id: str
    normal_login_hours: tuple[int, ...]
    home_countries: tuple[str, ...]
    known_device_ids: tuple[str, ...]
    known_source_ids: tuple[str, ...]
    preferred_auth_methods: tuple[AuthMethod, ...]
    baseline_success_rate: float


@dataclass(frozen=True, slots=True)
class SourceProfile:
    """Network-origin descriptor (represents an IP address or subnet).

    Attributes
    ----------
    source_id:
        Prefixed pseudonym ``s:<hex32>`` generated via UUIDv5.
    country_code:
        ISO 3166-1 alpha-2 country code for this origin.
    coarse_latitude:
        Approximate latitude in degrees (-90 to 90).
    coarse_longitude:
        Approximate longitude in degrees (-180 to 180).
    """

    source_id: str
    country_code: str
    coarse_latitude: float
    coarse_longitude: float


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """Device descriptor for synthetic authentication events.

    Attributes
    ----------
    device_id:
        Prefixed pseudonym ``d:<hex32>`` generated via UUIDv5.
    user_agent_family:
        Browser or client family (e.g. ``"Chrome"``).
    operating_system_family:
        OS family (e.g. ``"Windows"``).
    client_type:
        ``ClientType`` enum value.
    """

    device_id: str
    user_agent_family: str
    operating_system_family: str
    client_type: ClientType


@dataclass(frozen=True, slots=True)
class ApplicationProfile:
    """Synthetic application that receives authentication requests.

    Attributes
    ----------
    application_id:
        Short label such as ``"app-00"`` (no prefix required by schema).
    """

    application_id: str
