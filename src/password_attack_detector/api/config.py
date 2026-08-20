"""Typed serving configuration.

The split this module exists to enforce: **artifact location may vary, and
scientific identity may not.**

Every field below answers "where do I find things, and how much may a client
ask for" -- a directory, a port, a log level, a batch ceiling.  Not one of them
answers "which model, at what threshold, fused how".  Those were decided on
validation evidence and frozen; a deployment that could re-answer them from an
environment variable would be publishing a different system under the same name,
and it would do so without leaving a record anywhere.

:data:`PROHIBITED_SETTING_NAMES` lists the names that would cross that line, and
:func:`_assert_no_scientific_override_field` fails at import if one is ever
declared.  A test pins it too, so the guard cannot be quietly deleted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from password_attack_detector.exceptions import ConfigurationError

__all__ = [
    "DEFAULT_MAX_BATCH_EVENTS",
    "MAX_MAX_BATCH_EVENTS",
    "PROHIBITED_SETTING_NAMES",
    "APISettings",
    "load_api_settings",
]

#: The serving default for how many authentication events one request may carry.
#: Large enough for a demonstrable window of traffic, small enough that a single
#: request cannot occupy the process for an unbounded time.
DEFAULT_MAX_BATCH_EVENTS: Final[int] = 500

#: The hard ceiling an operator may raise ``max_batch_events`` to.  A limit an
#: operator could set to "unlimited" is not a limit.
MAX_MAX_BATCH_EVENTS: Final[int] = 5_000

#: The default request-body ceiling, in bytes.  Enforced by the serving layer
#: rather than assumed of whatever proxy happens to sit in front of it.
DEFAULT_MAX_REQUEST_BYTES: Final[int] = 1_048_576

#: The hard ceiling an operator may raise ``max_request_bytes`` to.
MAX_MAX_REQUEST_BYTES: Final[int] = 16_777_216

#: Setting names that would let a deployment restate a frozen scientific
#: decision.  None of them is declared, and none may be.
PROHIBITED_SETTING_NAMES: Final[frozenset[str]] = frozenset(
    {
        "model_id",
        "model_family",
        "catalog_model_id",
        "champion_model_id",
        "decision_threshold",
        "threshold",
        "ml_threshold",
        "min_detection_rate",
        "max_false_positive_rate",
        "fusion_strategy",
        "selected_fusion_strategy",
        "calibration",
        "calibrator",
        "score_kind",
        "feature_names",
        "eligible_features",
        "risk_score",
    }
)

_VALID_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


class APISettings(BaseSettings):
    """Deployment-side serving settings, and nothing scientific.

    Loaded from ``PAD_API_*`` environment variables, an untracked ``.env`` file,
    or explicit constructor arguments -- in that priority order, which is
    pydantic-settings' default and is deliberately not customised here.

    Paths are optional.  When one is omitted the runtime reports the component
    that needed it as unconfigured rather than guessing a location: a serving
    process that silently found "some artifacts" under a default directory would
    be serving a model nobody named.
    """

    model_config = SettingsConfigDict(
        env_prefix="PAD_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # -- process -----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65_535)
    log_level: str = "INFO"
    environment: Literal["development", "testing", "production"] = "development"
    #: Whether the interactive OpenAPI documents are served.  Left on by
    #: default for the demonstration; an operator who does not want a schema
    #: browser on a public port turns it off here.
    docs_enabled: bool = True

    # -- where the frozen artifacts live -----------------------------------
    #: Root holding ``champion/``, ``runs/``, ``ledger/`` and ``evaluations/``.
    artifact_root: Path | None = None
    #: The reviewed ML feature allowlist the champion was resolved against.
    allowlist_path: Path | None = None
    #: The Phase 3 feature configuration the champion's catalog was built from.
    feature_config_path: Path | None = None
    #: The ML configuration the champion was produced under.
    ml_config_path: Path | None = None
    #: The Phase 4 rule configuration this deployment runs.
    detection_config_path: Path | None = None
    #: Where the materialized serving bundles live.
    #:
    #: A **location**, and only a location.  It says where to find the published
    #: hybrid state; it cannot say which strategy to run, because the strategy is
    #: read out of the bundle's own frozen selection and checked against the
    #: locked evaluation receipt.  Pointing this at a different directory can
    #: therefore make a hybrid unavailable, and can never make a different one
    #: appear.  Unset, it defaults to ``serving/`` under the artifact root.
    serving_bundle_root: Path | None = None
    #: Which frozen champion scope to serve.  Required when more than one is
    #: frozen under ``artifact_root``; this names *which* frozen champion, and
    #: cannot name anything that is not frozen.
    champion_scope_key: str | None = None

    # -- request ceilings ---------------------------------------------------
    max_batch_events: int = Field(
        default=DEFAULT_MAX_BATCH_EVENTS, ge=1, le=MAX_MAX_BATCH_EVENTS
    )
    max_request_bytes: int = Field(
        default=DEFAULT_MAX_REQUEST_BYTES, ge=1_024, le=MAX_MAX_REQUEST_BYTES
    )

    # -- demonstration replay ----------------------------------------------
    #: Whether the synthetic live/replay demonstration endpoints are served.
    #:
    #: A **facility** switch, not a scientific one.  Turning replay off removes
    #: a way of *watching* the detector and changes nothing about what the
    #: detector decides: the scenarios are fixed, they are scored through the
    #: same orchestration ``/api/v1/detect`` uses, and no replay setting can name
    #: a model, a threshold, or a strategy.
    replay_enabled: bool = True
    #: Whether readiness depends on the replay subsystem.
    #:
    #: False by default, and that default is the decision: replay is an optional
    #: demonstration facility sharing a process with a detection service, and a
    #: detector that refused to serve because its demo history could not
    #: initialise would have its priorities backwards.  A deployment that exists
    #: *only* to demonstrate can set this true and get the opposite behaviour.
    replay_required: bool = False

    # -- degraded operation -------------------------------------------------
    #: Whether a loadable ML champion is required for the service to be ready.
    #:
    #: This can only switch the model layer **off entirely**, which every
    #: response and the readiness document then say out loud.  It cannot select
    #: a different model, move a threshold, or change a fusion strategy, so it
    #: is a deployment control rather than a scientific one.
    require_ml_champion: bool = True

    @field_validator("log_level", mode="before")
    @classmethod
    def check_log_level(cls, value: object) -> str:
        """Normalise and validate the log level."""
        text = str(value).upper()
        if text not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, got {value!r}"
            )
        return text

    @field_validator(
        "artifact_root",
        "allowlist_path",
        "feature_config_path",
        "ml_config_path",
        "detection_config_path",
        "serving_bundle_root",
    )
    @classmethod
    def check_path(cls, value: Path | None) -> Path | None:
        """Reject a configured path that is not absolute or is not normalised.

        These paths come from the deployment's own environment, never from a
        request.  Requiring them absolute and free of ``..`` keeps a
        misconfigured relative path from resolving against whatever working
        directory the process happened to start in.
        """
        if value is None:
            return None
        if ".." in value.parts:
            raise ValueError("a configured path must not contain a '..' segment")
        if not value.is_absolute():
            raise ValueError("a configured path must be absolute")
        return value

    @property
    def configured_paths(self) -> tuple[Path, ...]:
        """Return every configured artifact location, for startup verification."""
        return tuple(
            path
            for path in (
                self.artifact_root,
                self.allowlist_path,
                self.feature_config_path,
                self.ml_config_path,
                self.detection_config_path,
                self.serving_bundle_root,
            )
            if path is not None
        )

    @property
    def bundle_root(self) -> Path | None:
        """Return where serving bundles are read from, or ``None`` if nowhere.

        Defaults to the artifact root, under which
        :func:`~password_attack_detector.deployment.bundle.bundle_directory`
        locates one bundle per champion scope.  An explicit
        ``serving_bundle_root`` overrides that location and nothing else.
        """
        if self.serving_bundle_root is not None:
            return self.serving_bundle_root
        return self.artifact_root


def load_api_settings(**overrides: Any) -> APISettings:
    """Load the serving settings, or raise a project configuration error.

    Constructor overrides win over the environment, which is what lets a test
    inject a temporary artifact root without touching the process environment.

    Raises:
        ConfigurationError: when any field fails validation.  The message
            carries the failure and never the value that caused it.
    """
    try:
        return APISettings(**overrides)
    except Exception as exc:  # pragma: no cover - re-raised with a stable type
        raise ConfigurationError(f"Invalid API configuration: {exc}") from None


def _assert_no_scientific_override_field() -> None:
    """Fail at import if a setting appears that could restate a frozen decision."""
    offending = sorted(set(APISettings.model_fields) & PROHIBITED_SETTING_NAMES)
    if offending:
        raise ValueError(
            f"APISettings declares setting(s) {offending} that would let a "
            f"deployment restate a frozen scientific decision; artifact location "
            f"may vary, scientific identity may not"
        )


_assert_no_scientific_override_field()
