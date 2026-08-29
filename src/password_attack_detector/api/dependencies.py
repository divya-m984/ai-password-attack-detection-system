"""How a route reaches the runtime, and the one gate every detection passes.

The runtime is built once, during startup, and stored on the application object.
Routes never construct it, never mutate it, and never reach around it: they
receive it through :func:`get_runtime`, which is the only place the attribute is
read.  That keeps "which state is this request using" a single answerable
question rather than a search.

:func:`require_ready_runtime` is the gate.  Every detection route depends on it,
so "the runtime is ready" is enforced structurally rather than repeated in each
handler where one could eventually be forgotten.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from password_attack_detector.api.errors import APIError, ErrorCode
from password_attack_detector.api.services import RuntimeState

__all__ = [
    "ReadyRuntime",
    "Runtime",
    "get_runtime",
    "require_ready_runtime",
]

#: The attribute the application factory stores the runtime under.
RUNTIME_ATTRIBUTE = "detection_runtime"


def get_runtime(request: Request) -> RuntimeState:
    """Return the runtime this application was started with.

    Raises:
        APIError: with :attr:`ErrorCode.RUNTIME_NOT_READY` when no runtime is
            present.  That state is only reachable if startup never ran, which is
            an operational failure rather than a client one -- but it is still
            reported through the ordinary envelope rather than as a traceback.
    """
    runtime: RuntimeState | None = getattr(request.app.state, RUNTIME_ATTRIBUTE, None)
    if runtime is None:
        raise APIError(ErrorCode.RUNTIME_NOT_READY)
    return runtime


def require_ready_runtime(
    runtime: Annotated[RuntimeState, Depends(get_runtime)],
) -> RuntimeState:
    """Return the runtime, or refuse because it cannot serve detection.

    Raises:
        APIError: with :attr:`ErrorCode.RUNTIME_NOT_READY`.  The refusal names no
            component and no path; ``GET /ready`` is where an operator finds out
            which component is missing, and that document carries stable reason
            codes rather than internal detail.
    """
    if not runtime.ready:
        raise APIError(ErrorCode.RUNTIME_NOT_READY)
    return runtime


#: The runtime, ready or not.  For the operational documents, which have to be
#: answerable precisely when things are broken.
Runtime = Annotated[RuntimeState, Depends(get_runtime)]

#: The runtime, guaranteed ready.  For the detection routes.
ReadyRuntime = Annotated[RuntimeState, Depends(require_ready_runtime)]
