"""Deterministic synthetic authentication-event generation.

Public API
----------
- ``SyntheticConfig``    — typed generation configuration with fingerprinting
- ``EntityPopulation``   — users, sources, devices, and applications
- ``build_entity_population`` — construct a population from config + RNG
- ``GenerationResult``   — immutable container returned by ``generate_dataset``
- ``generate_dataset``   — single entry-point for end-to-end generation

Synthetic identifiers are produced deterministically via UUIDv5 and never
require a real HMAC key.  The ``PseudonymService`` is only used for real-data
ingestion.
"""

from __future__ import annotations

from password_attack_detector.data.synthetic.config import (
    BotActivityParams,
    BruteForceParams,
    CampaignParameters,
    CredentialStuffingParams,
    DistributedBruteForceParams,
    EnabledScenarios,
    ImpossibleTravelParams,
    NovelAnomalyParams,
    PasswordSprayingParams,
    SyntheticConfig,
)
from password_attack_detector.data.synthetic.entities import (
    EntityPopulation,
    build_entity_population,
    make_device_id,
    make_session_id,
    make_source_id,
    make_user_id,
)
from password_attack_detector.data.synthetic.generator import (
    GenerationResult,
    generate_dataset,
)
from password_attack_detector.data.synthetic.profiles import (
    ApplicationProfile,
    DeviceProfile,
    SourceProfile,
    UserProfile,
)

__all__ = [
    "ApplicationProfile",
    "BotActivityParams",
    "BruteForceParams",
    "CampaignParameters",
    "CredentialStuffingParams",
    "DeviceProfile",
    "DistributedBruteForceParams",
    "EnabledScenarios",
    "EntityPopulation",
    "GenerationResult",
    "ImpossibleTravelParams",
    "NovelAnomalyParams",
    "PasswordSprayingParams",
    "SourceProfile",
    "SyntheticConfig",
    "UserProfile",
    "build_entity_population",
    "generate_dataset",
    "make_device_id",
    "make_session_id",
    "make_source_id",
    "make_user_id",
]
