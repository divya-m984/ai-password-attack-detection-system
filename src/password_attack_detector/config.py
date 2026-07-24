"""Typed configuration system for the Password Attack Detector.

Settings are loaded from the following sources in descending priority:

1. Explicit constructor kwargs (``init_settings``)
2. ``PAD_*`` environment variables
3. ``.env`` file (silently ignored if absent)
4. Environment-specific YAML (``configs/{environment}.yaml``)
5. Field defaults

Use :func:`load_settings` rather than instantiating :class:`Settings` directly.
That function resolves the active environment, loads the corresponding YAML
file, and binds the data to a closure-based settings source — avoiding any
shared mutable class state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import SecretStr, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from password_attack_detector.exceptions import ConfigurationError
from password_attack_detector.paths import find_repo_root, get_configs_dir

__all__ = ["Settings", "load_settings"]

_VALID_ENVIRONMENTS: frozenset[str] = frozenset(
    {"development", "testing", "production"}
)
_VALID_LOG_LEVELS: frozenset[str] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


class _YamlConfigSource(PydanticBaseSettingsSource):
    """A pydantic-settings source that serves pre-loaded YAML data."""

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        yaml_data: dict[str, Any],
    ) -> None:
        super().__init__(settings_cls)
        self._yaml_data = yaml_data

    def get_field_value(
        self,
        field: FieldInfo,
        field_name: str,
    ) -> tuple[Any, str, bool]:
        value = self._yaml_data.get(field_name)
        return value, field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._yaml_data


class Settings(BaseSettings):
    """Validated, immutable project settings.

    Instantiate via :func:`load_settings` to ensure the correct environment
    YAML is applied and all sources are wired in the right priority order.
    """

    model_config = SettingsConfigDict(
        env_prefix="PAD_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "Password Attack Detector"
    environment: Literal["development", "testing", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"
    random_seed: int = 42

    # Path fields default to None and are resolved in set_default_paths below.
    # Using None avoids calling find_repo_root() at class-definition time,
    # which would break tests that inject a custom repo root via start=.
    data_dir: Path | None = None
    artifacts_dir: Path | None = None
    models_dir: Path | None = None
    reports_dir: Path | None = None
    config_dir: Path | None = None

    # Placeholder for future secret fields — SecretStr fields are redacted
    # automatically by the show-config CLI command.
    # example_secret: SecretStr | None = None

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: object) -> str:
        s = str(v).upper()
        if s not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, got {v!r}"
            )
        return s

    @model_validator(mode="after")
    def set_default_paths(self) -> Settings:
        root = find_repo_root()
        if self.data_dir is None:
            self.data_dir = root / "data"
        if self.artifacts_dir is None:
            self.artifacts_dir = root / "artifacts"
        if self.models_dir is None:
            self.models_dir = root / "models"
        if self.reports_dir is None:
            self.reports_dir = root / "reports"
        if self.config_dir is None:
            self.config_dir = root / "configs"
        return self


def load_settings(environment: str | None = None) -> Settings:
    """Load and validate settings for the given environment.

    Parameters
    ----------
    environment:
        Target environment name (``development``, ``testing``, or
        ``production``).  When omitted, the value of the ``PAD_ENVIRONMENT``
        environment variable is used, falling back to ``development``.

    Returns
    -------
    Settings
        Fully validated settings instance.

    Raises
    ------
    ConfigurationError
        When *environment* is not one of the recognised values.
    pydantic.ValidationError
        When a settings value fails field-level validation.
    """
    env = environment or os.environ.get("PAD_ENVIRONMENT", "development")

    if env not in _VALID_ENVIRONMENTS:
        raise ConfigurationError(
            f"Invalid environment: {env!r}. "
            f"Must be one of: {', '.join(sorted(_VALID_ENVIRONMENTS))}."
        )

    yaml_path = get_configs_dir() / f"{env}.yaml"
    yaml_data: dict[str, Any] = {}
    if yaml_path.exists():
        with yaml_path.open() as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                yaml_data = loaded

    # Capture yaml_data in a closure so each call gets its own independent
    # source — no shared mutable state, safe for concurrent usage.
    captured_yaml: dict[str, Any] = yaml_data

    class _BoundSettings(Settings):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            # pydantic-settings declares the return type as a tuple of sources
            # but the classmethod signature uses a generic base type that mypy
            # cannot reconcile with the override here; the ignore is minimal.
            return (
                init_settings,
                env_settings,
                dotenv_settings,
                _YamlConfigSource(settings_cls, captured_yaml),
            )

    return _BoundSettings()


def is_secret_field(value: object) -> bool:
    """Return True if *value* is a pydantic SecretStr instance."""
    return isinstance(value, SecretStr)
