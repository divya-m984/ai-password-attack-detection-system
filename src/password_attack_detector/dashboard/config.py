"""Typed dashboard configuration: where the API is, and how it is displayed.

The same split the serving layer enforces, applied one layer further out and for
a sharper reason.  :mod:`password_attack_detector.api.config` says *artifact
location may vary, scientific identity may not*.  Here the line is drawn tighter
still: this process holds no artifact, loads no model, and reads no threshold, so
its configuration answers only **which service to ask and how to render the
answer**.

That matters because a dashboard is the most tempting place in a system to add a
"quick override".  A model picker, a threshold slider, a fusion-strategy dropdown
-- each reads as a harmless convenience, and each would produce screenshots of a
system nobody deployed.  :data:`PROHIBITED_SETTING_NAMES` names the settings that
would cross that line, and :func:`_assert_no_scientific_override_field` fails at
import if one is ever declared.  A test pins the list too, so the guard cannot be
quietly deleted.

There is deliberately no ``api_key``, ``token``, or ``secret`` field either.  The
service this talks to accepts no credential material, so a credential here would
be one with nowhere to go and a file to leak out of.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from password_attack_detector.exceptions import ConfigurationError

__all__ = [
    "ALLOWED_API_SCHEMES",
    "DEFAULT_API_URL",
    "MAX_REFRESH_SECONDS",
    "MAX_REQUEST_TIMEOUT_SECONDS",
    "MIN_REFRESH_SECONDS",
    "PROHIBITED_SETTING_NAMES",
    "DashboardSettings",
    "load_dashboard_settings",
]

#: Where a local two-terminal demo finds the API.  A loopback address, because a
#: default pointing anywhere else would be a dashboard that talks to a machine
#: the operator did not name.
DEFAULT_API_URL: Final[str] = "http://127.0.0.1:8000"

#: Schemes the client will call.  Anything else -- ``file:``, ``ftp:``, a bare
#: path -- is a configuration error rather than something to attempt.
ALLOWED_API_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})

#: The ceiling on the per-request timeout.  A timeout an operator could set to
#: "wait forever" is not a timeout: a hung backend would hang every page with it.
MAX_REQUEST_TIMEOUT_SECONDS: Final[float] = 60.0

#: Bounds on the health-refresh interval.  The lower bound is what keeps a
#: refresh control from becoming a load generator pointed at the API.
MIN_REFRESH_SECONDS: Final[int] = 5
MAX_REFRESH_SECONDS: Final[int] = 3_600

#: Setting names that would let the dashboard restate a frozen scientific
#: decision, or hold a secret.  None of them is declared, and none may be.
PROHIBITED_SETTING_NAMES: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "artifact_root",
        "calibration",
        "calibrator",
        "catalog_model_id",
        "champion_model_id",
        "champion_scope_key",
        "decision_threshold",
        "eligible_features",
        "feature_names",
        "fusion_strategy",
        "min_detection_rate",
        "ml_threshold",
        "model_family",
        "model_id",
        "password",
        "risk_score",
        "score_kind",
        "secret",
        "selected_fusion_strategy",
        "serving_bundle_root",
        "threshold",
        "token",
    }
)


class DashboardSettings(BaseSettings):
    """Presentation-side settings, and nothing scientific.

    Loaded from ``PAD_DASHBOARD_*`` environment variables, an untracked ``.env``
    file, or explicit constructor arguments -- in that priority order, which is
    pydantic-settings' default and is deliberately not customised here.
    """

    model_config = SettingsConfigDict(
        env_prefix="PAD_DASHBOARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    #: Base URL of the serving API. A location, and only a location: it can make
    #: the dashboard talk to a different deployment, and can never make a
    #: deployment report a different model, threshold, or strategy.
    api_url: str = DEFAULT_API_URL
    #: How long any single API call may take before it is abandoned. Bounded on
    #: both sides: a page that waits indefinitely is a page that never renders.
    request_timeout_seconds: float = Field(default=5.0, gt=0.0)
    #: How often the manual refresh control offers to re-read health and status.
    #: Offered, never automatic -- see ``components.status``.
    refresh_seconds: int = Field(default=30, ge=MIN_REFRESH_SECONDS)
    page_title: str = Field(
        default="AI-Powered Password Attack Detection System",
        min_length=1,
        max_length=120,
    )

    @field_validator("api_url")
    @classmethod
    def check_api_url(cls, value: str) -> str:
        """Require an absolute http(s) URL with a host, and normalise it.

        Checked here rather than at the first request so a typo is a startup
        error naming the field, instead of a connection failure that reads like
        the API being down.
        """
        text = value.strip().rstrip("/")
        parts = urlsplit(text)
        if parts.scheme.lower() not in ALLOWED_API_SCHEMES:
            raise ValueError(
                f"api_url must use one of {sorted(ALLOWED_API_SCHEMES)}, "
                f"got {parts.scheme!r}"
            )
        if not parts.netloc:
            raise ValueError("api_url must name a host")
        if parts.query or parts.fragment:
            raise ValueError("api_url is a base URL and carries no query or fragment")
        return text

    @field_validator("request_timeout_seconds")
    @classmethod
    def check_timeout(cls, value: float) -> float:
        """Reject an unbounded or non-finite timeout."""
        if value > MAX_REQUEST_TIMEOUT_SECONDS:
            raise ValueError(
                f"request_timeout_seconds must not exceed {MAX_REQUEST_TIMEOUT_SECONDS}"
            )
        return value

    @field_validator("refresh_seconds")
    @classmethod
    def check_refresh(cls, value: int) -> int:
        """Reject a refresh interval that would make the dashboard a load source."""
        if value > MAX_REFRESH_SECONDS:
            raise ValueError(f"refresh_seconds must not exceed {MAX_REFRESH_SECONDS}")
        return value

    def endpoint(self, path: str) -> str:
        """Return the absolute URL for an API *path*.

        The single place a URL is assembled, so every request is provably built
        from the configured base rather than from something a page passed in.
        """
        return f"{self.api_url}/{path.lstrip('/')}"

    @property
    def display_api_url(self) -> str:
        """Return the API location as it may be shown to a viewer.

        Userinfo is stripped. The settings schema declares no credential field,
        so nothing should ever put one in a URL -- but a URL is exactly the kind
        of string somebody eventually pastes a token into, and this is the one
        place it would be rendered onto a page.
        """
        parts = urlsplit(self.api_url)
        host = parts.hostname or ""
        if parts.port is not None:
            host = f"{host}:{parts.port}"
        return f"{parts.scheme}://{host}{parts.path}"


def load_dashboard_settings(**overrides: Any) -> DashboardSettings:
    """Load the dashboard settings, or raise a project configuration error.

    The message is assembled from the failing **field names and rules**, never
    from ``str(ValidationError)``.  Pydantic appends ``input_value=...`` to its
    string form, and the app renders a configuration failure onto the page: a
    misconfigured ``api_url`` with a token pasted into its userinfo would put
    that token in a browser. So the offending value is dropped rather than
    scrubbed -- a scrubber has to be right every time.

    Raises:
        ConfigurationError: when any field fails validation.  The message names
            which field and which rule, and never the value that broke it.
    """
    try:
        return DashboardSettings(**overrides)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or '(model)'}: {item['msg']}"
            for item in exc.errors()
        )
        raise ConfigurationError(
            f"Invalid dashboard configuration: {problems}"
        ) from None
    except Exception:  # pragma: no cover - re-raised with a stable type
        raise ConfigurationError("Invalid dashboard configuration") from None


def _assert_no_scientific_override_field(names: Iterable[str] | None = None) -> None:
    """Fail at import if a setting appears that could restate a frozen decision.

    *names* defaults to this module's own declared fields, which is how it is
    called below.  It is a parameter so the guard can be exercised against a
    field set that *does* offend -- a guard nobody has ever seen fire is a guard
    nobody knows works.
    """
    declared = set(DashboardSettings.model_fields) if names is None else set(names)
    offending = sorted(declared & PROHIBITED_SETTING_NAMES)
    if offending:
        raise ValueError(
            f"DashboardSettings declares setting(s) {offending}; a dashboard "
            f"chooses how a system is displayed and never which system it is"
        )


_assert_no_scientific_override_field()
