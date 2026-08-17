"""The detection endpoints.

Both routes take a **window**: an ordered batch of authentication events plus a
declaration of which of them the caller wants a verdict for.  A window rather
than a bare event, because almost everything the rules and the model read is a
windowed or sequence quantity over the anchor's own strictly-prior history.  A
one-event request would produce a snapshot whose history is empty -- and an empty
history is not "unknown", it is *wrong*: it would score a brute-force burst's
tenth failure exactly like a first login of the day.

So the caller supplies the history, this service fabricates none, and the two
endpoints differ only in how many anchors they answer.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status

from password_attack_detector.api.dependencies import ReadyRuntime
from password_attack_detector.api.errors import ErrorResponse
from password_attack_detector.api.schemas import (
    BatchDetectionResponse,
    DetectionBatchRequest,
    DetectionResponse,
    DetectionWindowRequest,
)
from password_attack_detector.api.services import detect_batch, detect_single

__all__ = ["detection_router"]

detection_router = APIRouter(prefix="/api/v1", tags=["Detection"])

_FAILURES: dict[int | str, dict[str, Any]] = {
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "model": ErrorResponse,
        "description": "The window carries more events or bytes than accepted.",
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": "The request is malformed, or an event is not valid.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "A required detection layer is not available.",
    },
}


@detection_router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Score one anchor within a window",
    description=(
        "Evaluates the rule layer, the frozen model, and the frozen fusion "
        "strategy over one anchor, using the supplied window as its "
        "point-in-time history. By default the anchor is the window's latest "
        "event. The rule layer's risk_score is an ordinal 0-100 severity "
        "magnitude and is never combined arithmetically with the model's "
        "probability."
    ),
    responses=_FAILURES,
)
def detect(request: DetectionWindowRequest, runtime: ReadyRuntime) -> DetectionResponse:
    """Return the three-layer verdict for the single selected anchor."""
    return detect_single(runtime, request)


@detection_router.post(
    "/detect/batch",
    response_model=BatchDetectionResponse,
    summary="Score many anchors within one window",
    description=(
        "Evaluates the same three layers over every selected anchor in one "
        "validated, ordered batch. The window is bounded by this deployment's "
        "max_batch_events, reported by /api/v1/system/status. Anchors are "
        "returned in canonical (event time, event id) order, so the response "
        "does not depend on the order they were requested in."
    ),
    responses=_FAILURES,
)
def detect_many(
    request: DetectionBatchRequest, runtime: ReadyRuntime
) -> BatchDetectionResponse:
    """Return the three-layer verdict for every selected anchor."""
    return detect_batch(runtime, request)
