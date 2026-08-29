"""The attribution endpoint.

Takes the same window a detection takes, and answers a narrower question about
it: **which transformed columns moved the frozen model's decision for this
anchor, and by how much.**

Separate from ``/api/v1/detect`` rather than folded into its response, for two
reasons.  A detection verdict is what an alert is raised on and is wanted on
every call; an attribution is what an analyst opens afterwards for one row, and
attaching it to every verdict would put a per-column table behind every alert.
And the two can legitimately disagree about availability: a champion family with
no exact decomposition still produces a perfectly good verdict, so the
explanation reports itself unavailable while detection carries on.

Nothing here fits, calibrates, re-thresholds, or writes.  The decomposition is
Phase 5's own, called through the scope-free primitive that needs no claim about
which experimental population a live row belongs to.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from password_attack_detector.api.dependencies import ReadyRuntime
from password_attack_detector.api.errors import ErrorResponse
from password_attack_detector.api.schemas import (
    DetectionWindowRequest,
    ExplanationResponse,
)
from password_attack_detector.api.services import explain_document

__all__ = ["explain_router"]

explain_router = APIRouter(prefix="/api/v1", tags=["Detection"])


@explain_router.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="Attribute one anchor's model decision to transformed columns",
    description=(
        "Takes the same window ``/api/v1/detect`` takes and decomposes the "
        "frozen model's decision for the selected anchor over the transformed "
        "columns it read. The contributions sum to the decision function's own "
        "quantity -- a logit, a mean leaf score, or a threshold step -- and "
        "never to the calibrated probability. They are ranked by magnitude and "
        "bounded, and the response reports how many columns were omitted. A "
        "champion family with no exact decomposition reports the explanation "
        "unavailable with a reason rather than an approximation. Nothing is "
        "fitted and no operating point is read or changed."
    ),
    responses={
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
            "description": "The runtime or the frozen champion is not available.",
        },
    },
)
def explain(
    request: DetectionWindowRequest, runtime: ReadyRuntime
) -> ExplanationResponse:
    """Return the model-side attribution for the single selected anchor."""
    return explain_document(runtime, request)
