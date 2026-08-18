"""The dashboard's only door to the backend.

Every byte the dashboard displays about the detection system comes through this
module, and nothing here formats anything for a human: the client returns typed
results, :mod:`~password_attack_detector.dashboard.formatting` decides how they
look, and pages do neither.  Scattering ``httpx`` calls through page code is how
a UI ends up with five different behaviours for "the backend is down".

**A failure is a value, not an exception.**  Every method returns an
:class:`APIResult`, which either carries a parsed document or carries a
:class:`Problem` with a stable :class:`ProblemKind`.  Pages branch on the kind
and render a fixed message for it.  That is what makes the offline experience
uniform, and it is what keeps a traceback off the page: an exception escaping
into Streamlit is rendered, in full, in the browser.

**Nothing is invented.**  There is no cached last-good response standing in for a
live one, no default document for an endpoint that failed, and no zero
substituted for a count nobody returned. A page that cannot get an answer says
so.

**A detection is never retried.**  ``GET`` requests are safe to repeat and are
not repeated here either -- the retry a user wants is the one they asked for by
pressing a button. ``POST`` is worse than merely unhelpful to retry: a request
that timed out may well have been evaluated, and re-sending it would double an
entry in the session's own alert history for no gain.

**A URL is never echoed back.**  ``httpx`` puts the full request URL in its
exception messages, and this module never forwards an exception message. What a
page may display is
:attr:`~password_attack_detector.dashboard.config.DashboardSettings.display_api_url`,
which is assembled from the configured host and port alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from password_attack_detector.dashboard.config import DashboardSettings
from password_attack_detector.dashboard.contracts import (
    BatchDetectionDocument,
    DetectionDocument,
    ExplanationDocument,
    HealthDocument,
    ModelInfoDocument,
    ReadinessDocument,
    RuleCatalogDocument,
    SystemStatusDocument,
    VersionDocument,
)

__all__ = [
    "PROBLEM_MESSAGES",
    "APIResult",
    "DashboardAPIClient",
    "Problem",
    "ProblemKind",
]

DocumentT = TypeVar("DocumentT", bound=BaseModel)


class ProblemKind(StrEnum):
    """Every way a call can fail, as far as a page needs to distinguish them.

    Stable: a page branches on these, and the *message* beside each one is prose
    that may be reworded.  Deliberately coarse -- a viewer choosing what to do
    next has four real options (wait, retry, fix the request, fix the
    deployment), and a taxonomy finer than that is a taxonomy nobody reads.
    """

    #: No connection could be made. The service is not running, or not there.
    OFFLINE = "offline"
    #: The connection was made and the deadline passed. Nothing is known about
    #: whether the request was evaluated, which is why POST is not retried.
    TIMEOUT = "timeout"
    #: The service answered and refused the request: malformed, over a limit, or
    #: carrying something it does not accept.
    REFUSED = "refused"
    #: The service is running but cannot serve detection yet.
    NOT_READY = "not_ready"
    #: The service failed while handling the request.
    SERVER_ERROR = "server_error"
    #: Something answered, and it was not this API.
    MALFORMED = "malformed"


#: The fixed text a page shows for each kind.  Fixed, because a message
#: interpolated from an exception is a message that eventually interpolates a
#: URL, a path, or a stack frame.
PROBLEM_MESSAGES: Final[dict[ProblemKind, str]] = {
    ProblemKind.OFFLINE: (
        "The detection API is not reachable. Start the service and retry."
    ),
    ProblemKind.TIMEOUT: (
        "The detection API did not answer within the configured timeout."
    ),
    ProblemKind.REFUSED: "The detection API refused this request.",
    ProblemKind.NOT_READY: (
        "The detection API is running but is not ready to serve detection."
    ),
    ProblemKind.SERVER_ERROR: ("The detection API failed while handling this request."),
    ProblemKind.MALFORMED: (
        "The response did not match the detection API's published contract."
    ),
}


@dataclass(frozen=True, slots=True)
class Problem:
    """A failed call, in terms a page can render without knowing why.

    :attr:`code` is the API's own stable error code (``API013`` and friends) when
    the service supplied one, and ``None`` otherwise.  :attr:`detail` is the
    aggregate context the service chose to publish -- counts and limits.  Neither
    is ever built here from an exception.
    """

    kind: ProblemKind
    #: Stable API error code, when the service returned its error envelope.
    code: str | None = None
    #: The service's own human-readable message. Absent unless it sent one.
    message: str | None = None
    #: Aggregate context the service published: counts, limits, field names.
    detail: Mapping[str, int | str] | None = None
    #: HTTP status, when there was a response at all.
    status_code: int | None = None

    @property
    def summary(self) -> str:
        """Return the line a page shows for this problem."""
        base = PROBLEM_MESSAGES[self.kind]
        if self.message:
            return f"{base} {self.message}"
        return base


@dataclass(frozen=True, slots=True)
class APIResult[T]:
    """Either a parsed document or a problem. Never both, and never neither."""

    document: T | None = None
    problem: Problem | None = None

    @property
    def ok(self) -> bool:
        """Return whether the call produced a document."""
        return self.document is not None

    def unwrap(self) -> T:
        """Return the document, for a caller that has already checked :attr:`ok`.

        Raises:
            RuntimeError: when there is no document.  A page should branch on
                :attr:`ok` rather than reach this; it exists so a mistake is a
                loud failure in a test rather than a ``None`` rendered as text.
        """
        if self.document is None:
            raise RuntimeError("this result carries a problem, not a document")
        return self.document


class DashboardAPIClient:
    """A typed, bounded, non-retrying client for the detection API.

    Holds one :class:`httpx.Client` so connections are pooled across a page
    render.  It is cheap to construct and safe to keep in Streamlit session
    state; :meth:`close` releases the pool.
    """

    def __init__(
        self,
        settings: DashboardSettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Build a client for *settings*.

        Args:
            settings: the dashboard configuration. Supplies the base URL and the
                timeout, and nothing else -- there is no scientific setting it
                could supply.
            transport: an ``httpx`` transport to use instead of the network.
                How the test suite drives every branch below against a real
                ASGI application without binding a port.
        """
        self._settings = settings
        self._client = httpx.Client(
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            transport=transport,
            # Redirects are not followed. The configured base URL names the
            # service; a redirect would let whatever answers move the dashboard
            # somewhere the operator did not configure.
            follow_redirects=False,
        )

    @property
    def settings(self) -> DashboardSettings:
        """Return the configuration this client was built from."""
        return self._settings

    def close(self) -> None:
        """Release the connection pool."""
        self._client.close()

    def __enter__(self) -> DashboardAPIClient:
        """Return this client, for use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release the connection pool on exit."""
        self.close()

    # -- reads --------------------------------------------------------------

    def health(self) -> APIResult[HealthDocument]:
        """``GET /health`` -- is the process up."""
        return self._get("/health", HealthDocument)

    def readiness(self) -> APIResult[ReadinessDocument]:
        """``GET /ready`` -- can it actually serve detection.

        A 503 here is a *document*, not a problem: the body is the readiness
        report, and the report is precisely what the page needs to show. Only a
        503 with no readable body degrades to a problem.
        """
        return self._get("/ready", ReadinessDocument, ok_statuses=(200, 503))

    def version(self) -> APIResult[VersionDocument]:
        """``GET /version`` -- the contract versions this build implements."""
        return self._get("/version", VersionDocument)

    def system_status(self) -> APIResult[SystemStatusDocument]:
        """``GET /api/v1/system/status`` -- which layers are running."""
        return self._get("/api/v1/system/status", SystemStatusDocument)

    def model_info(self) -> APIResult[ModelInfoDocument]:
        """``GET /api/v1/model/info`` -- the frozen champion's public identity."""
        return self._get("/api/v1/model/info", ModelInfoDocument)

    def rules(self) -> APIResult[RuleCatalogDocument]:
        """``GET /api/v1/rules`` -- the public rule catalog."""
        return self._get("/api/v1/rules", RuleCatalogDocument)

    # -- writes -------------------------------------------------------------

    def detect(
        self, events: Sequence[Mapping[str, Any]], *, anchor_selection: str = "last"
    ) -> APIResult[DetectionDocument]:
        """``POST /api/v1/detect`` -- score one anchor in a window.

        Not retried under any failure. See the module docstring.
        """
        return self._post(
            "/api/v1/detect",
            self._window(events, anchor_selection),
            DetectionDocument,
        )

    def detect_batch(
        self, events: Sequence[Mapping[str, Any]], *, anchor_selection: str = "all"
    ) -> APIResult[BatchDetectionDocument]:
        """``POST /api/v1/detect/batch`` -- score every selected anchor."""
        return self._post(
            "/api/v1/detect/batch",
            self._window(events, anchor_selection),
            BatchDetectionDocument,
        )

    def explain(
        self, events: Sequence[Mapping[str, Any]], *, anchor_selection: str = "last"
    ) -> APIResult[ExplanationDocument]:
        """``POST /api/v1/explain`` -- attribute one anchor's model decision."""
        return self._post(
            "/api/v1/explain",
            self._window(events, anchor_selection),
            ExplanationDocument,
        )

    @staticmethod
    def _window(
        events: Sequence[Mapping[str, Any]], anchor_selection: str
    ) -> dict[str, Any]:
        """Return the request envelope, carrying exactly what the caller built.

        The events are passed through unaltered. Nothing here adds a field,
        renames one, or fills a default: the API's schema is the authority on
        what a valid event is, and a client that pre-massaged a body would be a
        second, unreviewed opinion about it.
        """
        return {
            "events": [dict(item) for item in events],
            "anchor_selection": anchor_selection,
        }

    # -- transport ----------------------------------------------------------

    def _get(
        self,
        path: str,
        document: type[DocumentT],
        *,
        ok_statuses: tuple[int, ...] = (200,),
    ) -> APIResult[DocumentT]:
        """Perform one GET and parse it, or return why it could not be."""
        try:
            response = self._client.get(self._settings.endpoint(path))
        except httpx.TimeoutException:
            return APIResult(problem=Problem(ProblemKind.TIMEOUT))
        except httpx.HTTPError:
            # Every transport failure httpx distinguishes -- DNS, refused
            # connection, TLS, a protocol violation -- is one thing to a viewer:
            # the service is not answering. The exception message, which carries
            # the URL, is discarded rather than forwarded.
            return APIResult(problem=Problem(ProblemKind.OFFLINE))
        return self._parse(response, document, ok_statuses=ok_statuses)

    def _post(
        self, path: str, body: Mapping[str, Any], document: type[DocumentT]
    ) -> APIResult[DocumentT]:
        """Perform one POST and parse it, or return why it could not be.

        One attempt. A timed-out detection may already have been evaluated, and
        a client that resent it would produce two session alerts for one window.
        """
        try:
            response = self._client.post(self._settings.endpoint(path), json=dict(body))
        except httpx.TimeoutException:
            return APIResult(problem=Problem(ProblemKind.TIMEOUT))
        except httpx.HTTPError:
            return APIResult(problem=Problem(ProblemKind.OFFLINE))
        return self._parse(response, document)

    @classmethod
    def _parse(
        cls,
        response: httpx.Response,
        document: type[DocumentT],
        *,
        ok_statuses: tuple[int, ...] = (200,),
    ) -> APIResult[DocumentT]:
        """Turn one response into a typed document or a stable problem."""
        if response.status_code not in ok_statuses:
            return APIResult(problem=cls._problem(response))
        try:
            payload = response.json()
        except ValueError:
            return APIResult(problem=Problem(ProblemKind.MALFORMED, status_code=200))
        try:
            return APIResult(document=document.model_validate(payload))
        except ValidationError:
            # The body parsed as JSON and was not this contract. Reported as a
            # contract mismatch rather than raised: a pydantic error message
            # quotes the offending input, and the page it would land on is a
            # browser.
            return APIResult(
                problem=Problem(ProblemKind.MALFORMED, status_code=response.status_code)
            )

    @staticmethod
    def _problem(response: httpx.Response) -> Problem:
        """Classify a non-success response, reading the API's error envelope.

        The envelope is optional as far as this client is concerned. A 500 from
        a proxy in front of the service carries HTML, not an error object, and
        the status alone is enough to say what happened.
        """
        kind = (
            ProblemKind.NOT_READY
            if response.status_code == 503
            else ProblemKind.SERVER_ERROR
            if response.status_code >= 500
            else ProblemKind.REFUSED
        )
        code: str | None = None
        message: str | None = None
        detail: Mapping[str, int | str] | None = None
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, Mapping):
            error = body.get("error")
            if isinstance(error, Mapping):
                raw_code = error.get("code")
                raw_message = error.get("message")
                raw_detail = error.get("detail")
                code = raw_code if isinstance(raw_code, str) else None
                message = raw_message if isinstance(raw_message, str) else None
                if isinstance(raw_detail, Mapping):
                    detail = {
                        str(key): value
                        for key, value in raw_detail.items()
                        if isinstance(value, int | str)
                    }
        return Problem(
            kind=kind,
            code=code,
            message=message,
            detail=detail,
            status_code=response.status_code,
        )


def _assert_the_client_is_the_only_backend_boundary() -> None:
    """Fail at import if this module acquires a detection capability.

    The dashboard is a client. If a scorer, a rule engine, or a fusion function
    were reachable from here, "the dashboard does not duplicate scoring logic"
    would be a convention rather than a fact -- and the first page that needed a
    number the API does not publish would compute one.
    """
    import sys

    forbidden = {
        "DetectionEngine",
        "FeatureEngine",
        "FrozenChampion",
        "RiskScorer",
        "StackedFusionState",
        "apply_frozen_binary_decision",
        "fuse",
        "load_serving_bundle",
        "local_contributions",
        "predict_serving_binary",
    }
    offending = sorted(forbidden & set(vars(sys.modules[__name__])))
    if offending:
        raise ValueError(
            f"{__name__} imported {offending}; the dashboard displays what the "
            f"service decided and must not be able to decide anything itself"
        )


_assert_the_client_is_the_only_backend_boundary()
