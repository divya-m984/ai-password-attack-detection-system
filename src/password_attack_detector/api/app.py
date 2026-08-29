"""The application factory, its lifespan, and the failure surface.

Three things this module is responsible for, and each one is a decision worth
stating rather than a convention worth following.

**Nothing is loaded at import.**  Importing this module builds a FastAPI object
and registers routes; it does not open an artifact, verify a model, or read a
configuration file.  The runtime is assembled in the lifespan, so importing the
module in a test, a linter, or a documentation build costs nothing and cannot
fail because a deployment's artifacts are absent.

**Startup fails visibly, not fatally.**  :func:`~password_attack_detector.api
.services.build_runtime` never raises; a component that could not be initialised
is recorded with a stable reason code, readiness is false, and every detection
route refuses.  The process still answers ``/health``, ``/version`` and
``/ready``, which is what makes a broken deployment diagnosable from a container
log instead of a crash loop.

**No traceback reaches a client.**  Four handlers cover the whole failure
surface, and every one of them renders the same envelope with a stable code.  The
catch-all logs the exception *type* and discards its message: a project exception
can legitimately name a column, a fingerprint, or a directory, and a response
body is not a private place.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from password_attack_detector import __version__
from password_attack_detector.api.config import APISettings, load_api_settings
from password_attack_detector.api.dependencies import RUNTIME_ATTRIBUTE
from password_attack_detector.api.errors import (
    APIError,
    ErrorCode,
    rejection_code,
    sanitize,
)
from password_attack_detector.api.routes import (
    detection_router,
    explain_router,
    health_router,
    replay_router,
    system_router,
)
from password_attack_detector.api.services import RuntimeState, build_runtime
from password_attack_detector.logging_config import get_logger, setup_logging

__all__ = ["RequestSizeLimitMiddleware", "app", "create_app"]

_log = get_logger(__name__)

API_TITLE = "Password Attack Detector API"
API_DESCRIPTION = """
A defensive authentication-anomaly detection service.

This API is an **adapter** over a detection system that was built, validated and
frozen before the service existed. It does not train, select, promote, calibrate,
or retune anything, and it exposes no endpoint that could: the model, its
operating point, the rule thresholds and the hybrid fusion strategy are all
frozen artifacts, and no request field can name or change any of them.

**Detection takes a window, not a lone event.** Nearly every signal the rules and
the model read is a windowed or sequence quantity over an event's strictly-prior
history, so a request supplies an ordered batch of events plus the anchors it
wants a verdict for. No history is fabricated for a caller who does not supply
one.

**Three layers, kept apart.** A response reports the rule verdict, the model
verdict and the fused verdict in separate objects. The rule layer's `risk_score`
is an ordinal 0-100 severity magnitude and the model's `probability` is a
likelihood; they are never averaged, weighted or blended, because the result
would have units nobody defined.

**Credentials are refused.** This service never accepts a password, a hash, a
token, or any other secret, under any spelling. Source addresses, where supplied,
are pseudonymized on arrival and never returned.
""".strip()

_TAGS: list[dict[str, Any]] = [
    {
        "name": "Health",
        "description": "Liveness, readiness, and the contract versions this build implements.",
    },
    {
        "name": "Detection",
        "description": "Score authentication windows against the frozen detection system.",
    },
    {
        "name": "System",
        "description": "Public-safe system, model, and rule-catalog information.",
    },
    {
        "name": "Demo",
        "description": (
            "Replay a built-in synthetic scenario through the same detection "
            "path as /api/v1/detect, one event at a time. Scenarios are "
            "fabricated, credential-free, and cannot be uploaded; runs are "
            "process-local and are not retained across a restart."
        ),
    },
]


class RequestSizeLimitMiddleware:
    """Refuse a request body larger than this deployment accepts.

    Enforced here rather than assumed of whichever proxy happens to sit in front
    of the process: a service whose only body limit lives in someone else's
    configuration has no body limit when it is run any other way.

    Both forms are covered.  A declared ``Content-Length`` over the ceiling is
    refused before the body is read at all.  A body that arrives without one, or
    that under-declares its length, is drawn here and counted as it arrives, and
    the refusal is sent the moment the total crosses.

    The body is buffered rather than merely counted on its way past.  Aborting
    mid-stream by raising would surface as whatever the framework makes of a
    failed body read -- a generic parse error, with somebody else's status code
    -- and the ceiling would stop being visible as a ceiling.  Buffering is safe
    precisely because this is the thing that bounds it: nothing over the limit is
    ever held.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass the request through, refusing it if the body is over the ceiling."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            await self._refuse(scope, receive, send)
            return

        buffered: list[Message] = []
        total = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                break
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                await self._refuse(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        replay = iter(buffered)

        async def buffered_receive() -> Message:
            """Replay what was drawn, then defer to the original transport."""
            try:
                return next(replay)
            except StopIteration:
                return await receive()

        await self.app(scope, buffered_receive, send)

    async def _refuse(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Send the payload-too-large envelope, naming the ceiling that applied."""
        error = APIError(
            ErrorCode.PAYLOAD_TOO_LARGE, detail={"max_request_bytes": self.max_bytes}
        )
        response = JSONResponse(content=error.as_dict(), status_code=error.status_code)
        await response(scope, receive, send)


def _content_length(scope: Scope) -> int | None:
    """Return the declared body length, or ``None`` when there is not one."""
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _error_response(error: APIError) -> JSONResponse:
    """Render an :class:`APIError` as the service's single failure envelope."""
    return JSONResponse(content=error.as_dict(), status_code=error.status_code)


async def _handle_api_error(request: Request, exc: Exception) -> JSONResponse:
    """Render a deliberate refusal, with the code the raiser chose."""
    assert isinstance(exc, APIError)  # registered for this type only
    _log.info(
        "request refused",
        code=str(exc.code),
        path=request.url.path,
        method=request.method,
    )
    return _error_response(exc)


async def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Render a schema failure without echoing any submitted value.

    Pydantic's error list carries an ``input`` field holding the value that
    failed, which for this service could be an identifier or -- in the case it
    most matters -- a credential somebody tried to send. So the list is used for
    exactly two things: counting the problems, and reading back the code a
    schema validator claimed. Not one byte of the submitted values is forwarded.
    """
    assert isinstance(exc, RequestValidationError)  # registered for this type only
    errors = exc.errors()
    claimed = rejection_code(str(item.get("msg", "")) for item in errors)
    code = claimed if claimed is not None else ErrorCode.MALFORMED_REQUEST
    _log.info(
        "request rejected by schema validation",
        code=str(code),
        path=request.url.path,
        problem_count=len(errors),
    )
    # A credential refusal reports no count: the number of prohibited field names
    # a request carried is a fact about the credentials it carried.
    detail: dict[str, int | str] | None = (
        None
        if code is ErrorCode.CREDENTIAL_FIELD_REJECTED
        else {"problem_count": len(errors)}
    )
    return _error_response(APIError(code, detail=detail))


async def _handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    """Render a routing failure in the same envelope as everything else."""
    assert isinstance(exc, StarletteHTTPException)  # registered for this type only
    code = (
        ErrorCode.NOT_FOUND
        if exc.status_code in (404, 405)
        else ErrorCode.MALFORMED_REQUEST
        if exc.status_code < 500
        else ErrorCode.INTERNAL_ERROR
    )
    error = APIError(code)
    # Starlette chose the status for a routing failure; keeping it means a 405
    # stays a 405 rather than becoming this code's default.
    error.status_code = exc.status_code
    return _error_response(error)


async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """Render anything unforeseen, logging the type and discarding the message."""
    _log.error(
        "unhandled exception while serving a request",
        path=request.url.path,
        method=request.method,
        **sanitize(exc),
    )
    return _error_response(APIError(ErrorCode.INTERNAL_ERROR))


def create_app(
    *,
    settings: APISettings | None = None,
    runtime: RuntimeState | None = None,
) -> FastAPI:
    """Build the serving application.

    Args:
        settings: the serving configuration.  Loaded from the environment when
            omitted.
        runtime: a pre-built runtime to serve.  Supplying one skips artifact
            resolution entirely, which is how a test drives the whole HTTP
            surface against a known runtime -- including a deliberately broken
            one -- without a filesystem or a trained model.

    The returned application loads nothing until it is started.
    """
    resolved = settings if settings is not None else load_api_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Assemble the immutable runtime once, before the first request."""
        setup_logging(resolved.log_level, resolved.environment)
        state = runtime if runtime is not None else build_runtime(resolved)
        setattr(application.state, RUNTIME_ATTRIBUTE, state)
        try:
            yield
        finally:
            # The detection runtime holds verified in-memory artifacts and no
            # connection or file handle, so there is nothing to release there.
            # The *replay* subsystem does hold background tasks, and a process
            # that exited with runs still marked ``running`` would leave its last
            # published state describing something that is not happening. Every
            # active run is cancelled and recorded as stopped first.
            if state.replay is not None and state.replay.engine is not None:
                await state.replay.engine.shutdown()
            # Cleared so a stopped application cannot answer from stale state.
            setattr(application.state, RUNTIME_ATTRIBUTE, None)

    application = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=__version__,
        openapi_tags=_TAGS,
        lifespan=lifespan,
        # Explicitly off: a debug application renders tracebacks into responses,
        # which is precisely what the error contract exists to prevent.
        debug=False,
        docs_url="/docs" if resolved.docs_enabled else None,
        redoc_url="/redoc" if resolved.docs_enabled else None,
        openapi_url="/openapi.json" if resolved.docs_enabled else None,
    )
    # No CORS middleware is installed. A browser origin that needs to call this
    # service will be named by the milestone that introduces it; a permissive
    # default would be a decision nobody made.
    application.add_middleware(
        RequestSizeLimitMiddleware, max_bytes=resolved.max_request_bytes
    )

    handlers: list[tuple[Any, Callable[[Request, Exception], Awaitable[JSONResponse]]]]
    handlers = [
        (APIError, _handle_api_error),
        (RequestValidationError, _handle_validation_error),
        (StarletteHTTPException, _handle_http_exception),
        (Exception, _handle_unexpected),
    ]
    for exception_class, handler in handlers:
        application.add_exception_handler(exception_class, handler)

    application.include_router(health_router)
    application.include_router(detection_router)
    application.include_router(explain_router)
    application.include_router(system_router)
    application.include_router(replay_router)
    if runtime is not None:
        # A test-injected runtime is available immediately, so the application
        # is usable without a lifespan for callers that do not start one.
        setattr(application.state, RUNTIME_ATTRIBUTE, runtime)
    return application


#: The application uvicorn serves:
#:
#:     uv run uvicorn password_attack_detector.api.app:app --host 127.0.0.1 --port 8000
#:
#: Constructing it registers routes and nothing else; every artifact is resolved
#: when the server starts the lifespan.
app = create_app()
