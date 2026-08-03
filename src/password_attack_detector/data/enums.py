"""Domain enumerations for authentication-event analysis.

All enums use ``str`` as the mixin base so they serialise to their string value
(not name) when included in Pydantic models, Parquet files, or JSON output.
This ensures stable serialisation across enum renames.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "AuthMethod",
    "AuthOutcome",
    "CampaignStage",
    "ClientType",
    "FailureReason",
    "MFAOutcome",
    "ScenarioType",
]


class AuthMethod(StrEnum):
    """Authentication mechanism used for the event."""

    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    MFA_SMS = "mfa_sms"
    MFA_EMAIL = "mfa_email"
    SSO = "sso"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    PASSKEY = "passkey"


class AuthOutcome(StrEnum):
    """Result of the authentication attempt.

    Consistency rules enforced by ``AuthEvent`` validators:

    - ``SUCCESS``   → ``failure_reason`` must be ``None``
    - ``FAILURE``   → ``failure_reason`` must be a ``FailureReason`` value
      (use ``UNKNOWN`` when the specific reason is unavailable)
    - ``BLOCKED``   → ``failure_reason`` must be a blocking-valid ``FailureReason``
    - ``CHALLENGED``→ ``failure_reason`` must be ``None``
    """

    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"
    CHALLENGED = "challenged"


class FailureReason(StrEnum):
    """Reason for a non-successful authentication outcome.

    ``UNKNOWN`` is an explicit member used when the source system cannot provide
    a more specific reason.  It must not be confused with a missing value.
    """

    INVALID_CREDENTIALS = "invalid_credentials"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_DISABLED = "account_disabled"
    ACCOUNT_NOT_FOUND = "account_not_found"
    MFA_FAILED = "mfa_failed"
    MFA_EXPIRED = "mfa_expired"
    TOKEN_EXPIRED = "token_expired"
    IP_BLOCKED = "ip_blocked"
    RATE_LIMITED = "rate_limited"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    UNKNOWN = "unknown"


class MFAOutcome(StrEnum):
    """Result of the multi-factor authentication step."""

    PASSED = "passed"
    FAILED = "failed"
    BYPASSED = "bypassed"
    NOT_REQUIRED = "not_required"
    NOT_ENROLLED = "not_enrolled"


class ClientType(StrEnum):
    """Type of client that initiated the authentication request."""

    WEB_BROWSER = "web_browser"
    MOBILE_APP = "mobile_app"
    DESKTOP_APP = "desktop_app"
    API_CLIENT = "api_client"
    CLI_TOOL = "cli_tool"
    BOT = "bot"
    UNKNOWN = "unknown"


class ScenarioType(StrEnum):
    """Synthetic ground-truth scenario label.

    ``NOVEL_ANOMALY_HOLDOUT`` events have ``supervised_training_eligible=False``
    and are reserved for anomaly-detection evaluation only.
    """

    NORMAL = "normal"
    BRUTE_FORCE = "brute_force"
    PASSWORD_SPRAYING = "password_spraying"
    CREDENTIAL_STUFFING = "credential_stuffing"
    DISTRIBUTED_BRUTE_FORCE = "distributed_brute_force"
    ACCOUNT_TAKEOVER_INDICATOR = "account_takeover_indicator"
    IMPOSSIBLE_TRAVEL = "impossible_travel"
    BOT_ACTIVITY = "bot_activity"
    NOVEL_ANOMALY_HOLDOUT = "novel_anomaly_holdout"


class CampaignStage(StrEnum):
    """Stage within a synthetic attack campaign lifecycle."""

    SETUP = "setup"
    ACTIVE = "active"
    EXFIL = "exfil"
    CLEANUP = "cleanup"
