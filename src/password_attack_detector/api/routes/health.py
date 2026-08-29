"""Liveness, readiness, and version.

The three are separate on purpose and are never collapsed:

* ``/health`` says the process is up.  It reads nothing and checks nothing, so a
  probe hitting it every second costs a JSON serialisation and no more.
* ``/ready`` says whether this process can actually serve a detection.  It reads
  runtime state resolved once at startup; it does not re-verify artifacts per
  request, because a readiness probe that reloaded a model would be a denial of
  service with a green tick on it.
* ``/version`` says what contracts this build implements.  Deterministic across
  machines: no hostname, no path, no process identity, no start time.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from password_attack_detector import __version__
from password_attack_detector.api.dependencies import Runtime
from password_attack_detector.api.errors import ErrorResponse
from password_attack_detector.api.schemas import (
    HealthResponse,
    ReadinessResponse,
    ReadinessState,
    VersionResponse,
)
from password_attack_detector.api.services import readiness_document, version_document

__all__ = ["health_router"]

health_router = APIRouter(tags=["Health"])


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Process liveness",
    description=(
        "Reports that the process is running and able to answer. Performs no "
        "artifact, model, or filesystem check; use /ready to learn whether "
        "detection can actually be served."
    ),
)
def health() -> HealthResponse:
    """Return process liveness."""
    return HealthResponse(version=__version__)


@health_router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Detection readiness",
    description=(
        "Reports whether every required runtime component is available. Returns "
        "503 when it is not, with a stable reason code per component. Reason "
        "codes never carry a path, a message, or a stack trace."
    ),
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "At least one required component is unavailable.",
        }
    },
)
def ready(runtime: Runtime, response: Response) -> ReadinessResponse:
    """Return aggregate component readiness, and the matching HTTP status."""
    document = readiness_document(runtime)
    if document.status is not ReadinessState.READY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return document


@health_router.get(
    "/version",
    response_model=VersionResponse,
    summary="Contract versions",
    description=(
        "Reports the package version and every schema contract this build "
        "implements. Deterministic: nothing host-specific appears."
    ),
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse}},
)
def version() -> VersionResponse:
    """Return the package and contract versions."""
    return version_document()
