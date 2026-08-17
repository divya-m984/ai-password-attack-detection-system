"""Stable API error codes and the single envelope every failure is rendered in.

Two rules govern everything here.

**A code is a contract.**  ``API007`` means "the rule layer could not be
prepared" in this release and in every later one; a client may branch on it.  The
*message* beside it is human-facing prose that may be reworded, so nothing
machine-readable is ever encoded only in the message.

**A client learns what went wrong, never where.**  No message in this module
carries a filesystem path, an artifact identifier, a stack frame, a coefficient,
a pseudonym, or a credential.  :func:`sanitize` is the one place an internal
exception becomes client-visible text, and it discards the exception's own
message entirely rather than trying to scrub it -- a scrubber has to be right
every time, and a discard has to be right once.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ERROR_MESSAGES",
    "APIError",
    "ErrorBody",
    "ErrorCode",
    "ErrorResponse",
    "rejection_code",
    "rejection_marker",
    "sanitize",
]


class ErrorCode(StrEnum):
    """Every failure this service distinguishes.

    Stable once published.  A new failure mode takes a new member; an existing
    member never changes meaning.
    """

    #: The body is not valid JSON, or does not match the request schema.
    MALFORMED_REQUEST = "API001"
    #: The body parsed, but an event is not a valid canonical authentication
    #: event -- a bad timestamp, a malformed identifier, an impossible
    #: outcome/failure-reason pairing.
    INVALID_AUTHENTICATION_EVENT = "API002"
    #: Two events in one window claim the same ``event_id``.
    DUPLICATE_EVENT_IDENTITY = "API003"
    #: The events are not in non-decreasing ``event_time`` order.
    EVENT_ORDERING_ERROR = "API004"
    #: The window carries more events than this deployment admits.
    BATCH_LIMIT_EXCEEDED = "API005"
    #: The feature contract this build computes disagrees with the one the
    #: frozen champion was fitted under.
    FEATURE_CONTRACT_FAILURE = "API006"
    #: The rule engine could not be prepared or could not evaluate the window.
    RULE_DETECTION_UNAVAILABLE = "API007"
    #: The frozen ML champion could not be loaded, verified, or applied.
    ML_CHAMPION_UNAVAILABLE = "API008"
    #: No hybrid strategy is applicable to this deployment.
    FUSION_UNAVAILABLE = "API009"
    #: The runtime is not ready to serve detection.
    RUNTIME_NOT_READY = "API010"
    #: The request names an anchor the window does not contain, or selects a
    #: number of anchors the endpoint cannot answer.
    ANCHOR_SELECTION_ERROR = "API011"
    #: The request body exceeds the configured ceiling.
    PAYLOAD_TOO_LARGE = "API012"
    #: The request offers credential material, which this service never accepts.
    CREDENTIAL_FIELD_REJECTED = "API013"
    #: A source address was supplied but this deployment holds no
    #: pseudonymization key, so it cannot be processed under the privacy model.
    PSEUDONYMIZATION_UNAVAILABLE = "API014"
    #: No route matches, or the method is not allowed on this route.
    NOT_FOUND = "API015"
    #: Anything else.  Logged internally with context; reported without any.
    INTERNAL_ERROR = "API099"


#: The client-visible message for each code.  Deliberately fixed text: a message
#: interpolated from an exception is a message that eventually interpolates a
#: path.  Where a detail genuinely helps, it is passed explicitly as *detail*.
ERROR_MESSAGES: Final[dict[ErrorCode, str]] = {
    ErrorCode.MALFORMED_REQUEST: (
        "The request body is not valid for this endpoint's schema."
    ),
    ErrorCode.INVALID_AUTHENTICATION_EVENT: (
        "An authentication event in the window is not a valid canonical event."
    ),
    ErrorCode.DUPLICATE_EVENT_IDENTITY: (
        "Two events in the window declare the same event identity."
    ),
    ErrorCode.EVENT_ORDERING_ERROR: (
        "Events must be supplied in non-decreasing event_time order."
    ),
    ErrorCode.BATCH_LIMIT_EXCEEDED: (
        "The window carries more events than this deployment accepts."
    ),
    ErrorCode.FEATURE_CONTRACT_FAILURE: (
        "The runtime feature contract does not match the one the frozen model "
        "was fitted under."
    ),
    ErrorCode.RULE_DETECTION_UNAVAILABLE: (
        "The rule detection layer is not available."
    ),
    ErrorCode.ML_CHAMPION_UNAVAILABLE: (
        "The frozen machine-learning champion is not available."
    ),
    ErrorCode.FUSION_UNAVAILABLE: (
        "No hybrid fusion strategy is available for this deployment."
    ),
    ErrorCode.RUNTIME_NOT_READY: (
        "The detection runtime is not ready to serve requests."
    ),
    ErrorCode.ANCHOR_SELECTION_ERROR: (
        "The requested anchor selection cannot be answered by this endpoint."
    ),
    ErrorCode.PAYLOAD_TOO_LARGE: (
        "The request body exceeds the size this deployment accepts."
    ),
    ErrorCode.CREDENTIAL_FIELD_REJECTED: (
        "This service never accepts passwords, hashes, tokens, or secrets."
    ),
    ErrorCode.PSEUDONYMIZATION_UNAVAILABLE: (
        "A source address was supplied but this deployment cannot pseudonymize "
        "it; supply a pseudonymous source_id instead."
    ),
    ErrorCode.NOT_FOUND: "No such resource on this service.",
    ErrorCode.INTERNAL_ERROR: "The service could not complete the request.",
}

#: The HTTP status each code is reported with.
_STATUS: Final[dict[ErrorCode, int]] = {
    ErrorCode.MALFORMED_REQUEST: 422,
    ErrorCode.INVALID_AUTHENTICATION_EVENT: 422,
    ErrorCode.DUPLICATE_EVENT_IDENTITY: 422,
    ErrorCode.EVENT_ORDERING_ERROR: 422,
    ErrorCode.BATCH_LIMIT_EXCEEDED: 413,
    ErrorCode.FEATURE_CONTRACT_FAILURE: 503,
    ErrorCode.RULE_DETECTION_UNAVAILABLE: 503,
    ErrorCode.ML_CHAMPION_UNAVAILABLE: 503,
    ErrorCode.FUSION_UNAVAILABLE: 503,
    ErrorCode.RUNTIME_NOT_READY: 503,
    ErrorCode.ANCHOR_SELECTION_ERROR: 422,
    ErrorCode.PAYLOAD_TOO_LARGE: 413,
    ErrorCode.CREDENTIAL_FIELD_REJECTED: 422,
    ErrorCode.PSEUDONYMIZATION_UNAVAILABLE: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.INTERNAL_ERROR: 500,
}


class ErrorBody(BaseModel):
    """The error object itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ErrorCode = Field(description="Stable machine-readable failure code.")
    message: str = Field(description="Human-readable description of the failure.")
    #: Optional structured context, always sanitized and always aggregate:
    #: counts, field names, and limits.  Never a value, a path, or a row.
    detail: dict[str, int | str] | None = Field(
        default=None,
        description=(
            "Optional aggregate context: counts, limits, and field names only."
        ),
    )


class ErrorResponse(BaseModel):
    """The single envelope every failure on this service is rendered in."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    error: ErrorBody


class APIError(Exception):
    """A failure with a stable code, raised anywhere below the route layer.

    Carries no exception chain into the response: the cause is logged where the
    error is raised, and what reaches the client is the code, the fixed message
    for that code, and whatever aggregate detail was passed deliberately.
    """

    __slots__ = ("code", "detail", "message", "status_code")

    def __init__(
        self,
        code: ErrorCode,
        *,
        message: str | None = None,
        detail: dict[str, int | str] | None = None,
    ) -> None:
        self.code = code
        self.message = message if message is not None else ERROR_MESSAGES[code]
        self.detail = detail
        self.status_code = _STATUS[code]
        super().__init__(self.message)

    def as_response(self) -> ErrorResponse:
        """Return the envelope this error is rendered as."""
        return ErrorResponse(
            error=ErrorBody(code=self.code, message=self.message, detail=self.detail)
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the envelope as a JSON-serialisable mapping."""
        return self.as_response().model_dump(mode="json", exclude_none=True)


#: Matches the code token :func:`rejection_marker` embeds in a validation
#: message.  Anchored on the published code shape, so ordinary prose that
#: happens to contain brackets cannot be mistaken for one.
_MARKER_RE: Final[re.Pattern[str]] = re.compile(r"\[(API\d{3})\]")


def rejection_marker(code: ErrorCode) -> str:
    """Return the token a schema validator embeds to claim a specific code.

    A Pydantic validator can only raise a ``ValueError``, and by the time the
    resulting ``RequestValidationError`` reaches the handler its individual
    causes are indistinguishable prose.  Rather than pattern-matching on that
    prose -- which would make every message rewording a silent behaviour change
    -- a validator states the code it means, and :func:`rejection_code` reads it
    back.  The token is machine-readable and the surrounding wording is not
    load-bearing.
    """
    return f"[{code.value}]"


def rejection_code(messages: Iterable[str]) -> ErrorCode | None:
    """Return the code a validation message claimed, or ``None`` when none did.

    The request schemas are arranged so at most one marker can fire on a body --
    the credential scan runs before field validation, and the window checks stop
    at the first failure -- but the tie-break is written down rather than left to
    set iteration order.  A credential refusal outranks everything: if a request
    somehow both offered a secret and misordered its events, the secret is the
    part the caller needs to hear about.  Beyond that, the lowest code wins, so
    the answer never depends on which message happened to come first.

    A token naming a code this build does not define is ignored rather than
    raised on: this runs inside an exception handler, and a handler that can
    itself fail is a 500 waiting to happen.
    """
    found = {
        match.group(1) for message in messages for match in _MARKER_RE.finditer(message)
    }
    known = {code for code in ErrorCode if code.value in found}
    if not known:
        return None
    if ErrorCode.CREDENTIAL_FIELD_REJECTED in known:
        return ErrorCode.CREDENTIAL_FIELD_REJECTED
    return min(known, key=lambda item: item.value)


def sanitize(exc: BaseException) -> dict[str, str]:
    """Return the only thing an internal exception may contribute to a log line.

    The exception's *type* and nothing else.  A project exception message can
    legitimately name a column, a fingerprint, or a directory, and a log record
    is not a private place; the type is enough to route an investigation to the
    code that raised it.
    """
    return {"error_type": type(exc).__name__}
