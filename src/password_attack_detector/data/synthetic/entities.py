"""Entity population construction for synthetic dataset generation.

All identifiers are produced via UUIDv5 with fixed project-internal namespaces.
This guarantees:
- The same ``(seed, index)`` pair always produces the same ID.
- IDs from different namespaces (user/source/device/session) are guaranteed
  to be distinct even for the same index.
- No wall clock, process state, or machine identity is involved.

Forbidden ID sources
--------------------
``uuid.uuid4()``, ``time.time()``, ``os.getpid()``, ``random.random()``,
``hash()``, and any other non-deterministic source must **never** be used
for synthetic entity or event identifiers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from password_attack_detector.data.enums import AuthMethod, ClientType
from password_attack_detector.data.synthetic.config import SyntheticConfig
from password_attack_detector.data.synthetic.profiles import (
    ApplicationProfile,
    DeviceProfile,
    SourceProfile,
    UserProfile,
)

__all__ = [
    "EntityPopulation",
    "build_entity_population",
    "make_device_id",
    "make_session_id",
    "make_source_id",
    "make_user_id",
]

# ---------------------------------------------------------------------------
# Fixed project-internal UUID5 namespaces.
# These constants MUST NOT change between generator versions.
# ---------------------------------------------------------------------------
_NS_USER = uuid.UUID("c7d4a3f0-1b2e-4c8d-9e5f-6a7b8c9d0e1f")
_NS_SOURCE = uuid.UUID("d8e5b4a1-2c3f-4d9e-af60-7b8c9d0e1f2a")
_NS_DEVICE = uuid.UUID("e9f6c5b2-3d4a-4e0f-b071-8c9d0e1f2a3b")
_NS_SESSION = uuid.UUID("fa07d6c3-4e5b-4f10-c182-9d0e1f2a3b4c")

# ---------------------------------------------------------------------------
# Country-to-coarse-geo lookup table.
# ---------------------------------------------------------------------------
_COUNTRY_GEO: dict[str, tuple[float, float]] = {
    "AU": (-25.0, 133.0),
    "BR": (-10.0, -55.0),
    "CA": (60.0, -95.0),
    "CN": (35.0, 105.0),
    "DE": (51.0, 10.0),
    "FR": (46.0, 2.0),
    "GB": (55.0, -3.0),
    "IN": (20.0, 77.0),
    "JP": (36.0, 138.0),
    "KR": (36.0, 128.0),
    "MX": (23.0, -102.0),
    "NG": (10.0, 8.0),
    "NL": (52.0, 5.0),
    "NO": (62.0, 10.0),
    "PL": (52.0, 20.0),
    "RU": (60.0, 100.0),
    "SE": (62.0, 15.0),
    "SG": (1.0, 104.0),
    "US": (38.0, -97.0),
    "ZA": (-29.0, 25.0),
}
_COUNTRY_CODES: list[str] = sorted(_COUNTRY_GEO)

_USER_AGENTS: list[str] = ["Chrome", "Firefox", "Safari", "Edge", "curl"]
_OS_FAMILIES: list[str] = ["Windows", "macOS", "Linux", "iOS", "Android"]
_CLIENT_TYPES: list[ClientType] = [
    ClientType.WEB_BROWSER,
    ClientType.MOBILE_APP,
    ClientType.DESKTOP_APP,
    ClientType.API_CLIENT,
    ClientType.CLI_TOOL,
]
_AUTH_METHODS: list[AuthMethod] = [
    AuthMethod.PASSWORD,
    AuthMethod.MFA_TOTP,
    AuthMethod.MFA_EMAIL,
    AuthMethod.SSO,
    AuthMethod.PASSKEY,
]


# ---------------------------------------------------------------------------
# Public ID helpers
# ---------------------------------------------------------------------------


def make_user_id(seed: int, index: int) -> str:
    """Return a deterministic ``u:<hex32>`` pseudonym."""
    return f"u:{uuid.uuid5(_NS_USER, f'{seed}:{index}').hex}"


def make_source_id(seed: int, index: int) -> str:
    """Return a deterministic ``s:<hex32>`` pseudonym."""
    return f"s:{uuid.uuid5(_NS_SOURCE, f'{seed}:{index}').hex}"


def make_device_id(seed: int, index: int) -> str:
    """Return a deterministic ``d:<hex32>`` pseudonym."""
    return f"d:{uuid.uuid5(_NS_DEVICE, f'{seed}:{index}').hex}"


def make_session_id(seed: int, counter: int) -> str:
    """Return a deterministic ``sess:<hex32>`` pseudonym."""
    return f"sess:{uuid.uuid5(_NS_SESSION, f'{seed}:{counter}').hex}"


# ---------------------------------------------------------------------------
# EntityPopulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityPopulation:
    """Immutable set of synthetic entities for one generation run."""

    users: tuple[UserProfile, ...]
    sources: tuple[SourceProfile, ...]
    devices: tuple[DeviceProfile, ...]
    applications: tuple[ApplicationProfile, ...]


def build_entity_population(
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> EntityPopulation:
    """Build the full entity population for a generation run.

    Entity IDs are deterministic (UUIDv5 keyed on ``config.seed`` + index).
    The ``rng`` governs only behavioral attributes such as login-hour ranges,
    preferred auth methods, and baseline success rates.

    Parameters
    ----------
    config:
        Generation configuration; ``config.seed`` is used as the UUIDv5 key.
    rng:
        NumPy Generator instance (from ``np.random.default_rng(config.seed)``).
        Must be the generator's own RNG to preserve deterministic RNG state.

    Returns
    -------
    EntityPopulation
        Fully constructed population; immutable and safe to share.
    """
    applications = tuple(
        ApplicationProfile(application_id=f"app-{i:02d}")
        for i in range(config.num_applications)
    )

    # --- Sources ---
    sources_list: list[SourceProfile] = []
    for i in range(config.num_sources):
        country = _COUNTRY_CODES[i % len(_COUNTRY_CODES)]
        base_lat, base_lon = _COUNTRY_GEO[country]
        lat_j = float(rng.uniform(-2.0, 2.0))
        lon_j = float(rng.uniform(-2.0, 2.0))
        # Clamp to valid range after jitter
        lat = max(-90.0, min(90.0, round(base_lat + lat_j, 1)))
        lon = max(-180.0, min(180.0, round(base_lon + lon_j, 1)))
        sources_list.append(
            SourceProfile(
                source_id=make_source_id(config.seed, i),
                country_code=country,
                coarse_latitude=lat,
                coarse_longitude=lon,
            )
        )
    sources = tuple(sources_list)

    # --- Devices ---
    devices_list: list[DeviceProfile] = []
    for i in range(config.num_devices):
        devices_list.append(
            DeviceProfile(
                device_id=make_device_id(config.seed, i),
                user_agent_family=_USER_AGENTS[i % len(_USER_AGENTS)],
                operating_system_family=_OS_FAMILIES[i % len(_OS_FAMILIES)],
                client_type=_CLIENT_TYPES[i % len(_CLIENT_TYPES)],
            )
        )
    devices = tuple(devices_list)

    # --- Users ---
    devices_per_user = max(1, config.num_devices // config.num_users)
    sources_per_user = max(1, config.num_sources // config.num_users)

    users_list: list[UserProfile] = []
    for i in range(config.num_users):
        start_dev = (i * devices_per_user) % config.num_devices
        user_devices = tuple(
            devices[(start_dev + j) % config.num_devices].device_id
            for j in range(devices_per_user)
        )
        start_src = (i * sources_per_user) % config.num_sources
        user_sources = tuple(
            sources[(start_src + j) % config.num_sources].source_id
            for j in range(sources_per_user)
        )

        start_hour = int(rng.integers(0, 16))
        login_hours = tuple(range(start_hour, start_hour + 8))

        home_country = _COUNTRY_CODES[i % len(_COUNTRY_CODES)]

        n_methods = int(rng.integers(1, 3))
        perm = rng.permutation(len(_AUTH_METHODS))
        preferred_methods = tuple(_AUTH_METHODS[int(perm[j])] for j in range(n_methods))

        success_rate = float(rng.uniform(0.85, 0.99))

        users_list.append(
            UserProfile(
                user_id=make_user_id(config.seed, i),
                normal_login_hours=login_hours,
                home_countries=(home_country,),
                known_device_ids=user_devices,
                known_source_ids=user_sources,
                preferred_auth_methods=preferred_methods,
                baseline_success_rate=success_rate,
            )
        )

    return EntityPopulation(
        users=tuple(users_list),
        sources=sources,
        devices=devices,
        applications=applications,
    )
