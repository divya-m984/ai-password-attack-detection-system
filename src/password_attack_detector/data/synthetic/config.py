"""Typed configuration for synthetic dataset generation.

``SyntheticConfig`` is the single source of truth for one generation run.
Its ``fingerprint()`` method produces a SHA-256 hex string that identifies the
configuration deterministically, excluding output paths, overwrite flags,
secrets, current timestamps, and any other machine-specific values.

The same ``SyntheticConfig`` with the same ``seed`` always produces identical
output; changing any fingerprint-affecting field (including the seed) changes
the fingerprint and the generated data.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from password_attack_detector.data.schemas import SCHEMA_VERSION

__all__ = [
    "BotActivityParams",
    "BruteForceParams",
    "CampaignParameters",
    "CredentialStuffingParams",
    "DistributedBruteForceParams",
    "EnabledScenarios",
    "ImpossibleTravelParams",
    "NovelAnomalyParams",
    "PasswordSprayingParams",
    "SyntheticConfig",
]


class BruteForceParams(BaseModel):
    """Parameters for a single-source brute-force campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempts_per_campaign: int = Field(default=20, ge=1)
    num_campaigns: int = Field(default=1, ge=1)


class PasswordSprayingParams(BaseModel):
    """Parameters for a password-spraying campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passwords_per_round: int = Field(default=5, ge=1)
    num_campaigns: int = Field(default=1, ge=1)


class CredentialStuffingParams(BaseModel):
    """Parameters for a credential-stuffing campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    credentials_per_batch: int = Field(default=20, ge=1)
    num_campaigns: int = Field(default=1, ge=1)


class DistributedBruteForceParams(BaseModel):
    """Parameters for a distributed brute-force campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempts_per_source: int = Field(default=5, ge=1)
    num_sources: int = Field(default=5, ge=2)
    num_campaigns: int = Field(default=1, ge=1)


class AccountTakeoverParams(BaseModel):
    """Parameters for an account-takeover-indicator campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_campaigns: int = Field(default=1, ge=1)


class ImpossibleTravelParams(BaseModel):
    """Parameters for an impossible-travel campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_campaigns: int = Field(default=1, ge=1)


class BotActivityParams(BaseModel):
    """Parameters for a bot-activity campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    events_per_campaign: int = Field(default=30, ge=1)
    num_campaigns: int = Field(default=1, ge=1)


class NovelAnomalyParams(BaseModel):
    """Parameters for a novel-anomaly holdout campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_campaigns: int = Field(default=1, ge=1)


class CampaignParameters(BaseModel):
    """Per-scenario campaign configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    brute_force: BruteForceParams = Field(default_factory=BruteForceParams)
    password_spraying: PasswordSprayingParams = Field(
        default_factory=PasswordSprayingParams
    )
    credential_stuffing: CredentialStuffingParams = Field(
        default_factory=CredentialStuffingParams
    )
    distributed_brute_force: DistributedBruteForceParams = Field(
        default_factory=DistributedBruteForceParams
    )
    account_takeover_indicator: AccountTakeoverParams = Field(
        default_factory=AccountTakeoverParams
    )
    impossible_travel: ImpossibleTravelParams = Field(
        default_factory=ImpossibleTravelParams
    )
    bot_activity: BotActivityParams = Field(default_factory=BotActivityParams)
    novel_anomaly_holdout: NovelAnomalyParams = Field(
        default_factory=NovelAnomalyParams
    )


class EnabledScenarios(BaseModel):
    """Flags controlling which scenario generators run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    normal: bool = True
    brute_force: bool = True
    password_spraying: bool = True
    credential_stuffing: bool = True
    distributed_brute_force: bool = True
    account_takeover_indicator: bool = True
    impossible_travel: bool = True
    bot_activity: bool = True
    novel_anomaly_holdout: bool = True


class SyntheticConfig(BaseModel):
    """Complete configuration for one synthetic dataset generation run.

    All timing is relative to ``start_time``; the generator never reads the
    system clock.  The same ``SyntheticConfig`` + ``seed`` is guaranteed to
    produce bit-identical output across platforms and Python versions as long
    as the generator code is unchanged.

    ``fingerprint()`` returns a SHA-256 hex digest that uniquely identifies
    the generative configuration.  It excludes output paths, overwrite flags,
    secrets, current timestamps, and any machine-specific metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    generator_version: str = "1.0.0"
    schema_version: str = SCHEMA_VERSION
    seed: int = 42
    start_time: AwareDatetime
    duration_hours: int = Field(default=24, ge=1)
    num_users: int = Field(default=50, ge=1)
    num_sources: int = Field(default=20, ge=1)
    num_devices: int = Field(default=100, ge=1)
    num_applications: int = Field(default=3, ge=1)
    events_per_hour: int = Field(default=100, ge=1)
    enabled_scenarios: EnabledScenarios = Field(default_factory=EnabledScenarios)
    campaign_parameters: CampaignParameters = Field(default_factory=CampaignParameters)

    @field_validator("start_time", mode="after")
    @classmethod
    def normalize_to_utc(cls, v: datetime) -> datetime:
        """Normalise start_time to UTC."""
        return v.astimezone(UTC)

    def fingerprint(self) -> str:
        """Return a SHA-256 hex digest of the stable canonical config.

        The digest covers: generator_version, schema_version, seed,
        start_time (ISO-8601), duration_hours, entity counts,
        events_per_hour, enabled_scenarios, and campaign_parameters.

        Excluded: output paths, overwrite flags, secrets, current
        timestamps, and machine-specific values.
        """
        data: dict[str, Any] = {
            "generator_version": self.generator_version,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "start_time": self.start_time.isoformat(),
            "duration_hours": self.duration_hours,
            "num_users": self.num_users,
            "num_sources": self.num_sources,
            "num_devices": self.num_devices,
            "num_applications": self.num_applications,
            "events_per_hour": self.events_per_hour,
            "enabled_scenarios": self.enabled_scenarios.model_dump(),
            "campaign_parameters": self.campaign_parameters.model_dump(),
        }
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
