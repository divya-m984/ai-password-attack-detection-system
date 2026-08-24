#!/usr/bin/env python3
"""PID 1 for the single-container Render deployment.

Render's free tier gives one web service, one public port, no persistent disk,
and no orchestrator inside the container. The three processes that Compose runs
as three services on a VPS therefore have to share one container, and something
has to be their init. This is that something.

    Render $PORT  ->  caddy  ->  127.0.0.1:8501  Streamlit console
                          \\ ->  127.0.0.1:8000  FastAPI  ->  baked bundle

WHAT THIS IS, AND WHAT IT REFUSES TO BE

It starts three processes in a fixed order, waits for each to be genuinely
serving before starting the next, and then does nothing until something happens.
It is not a scheduler, not a restarter, and not a health manager: if any of the
three exits, for any reason, this process tears the other two down and exits
non-zero. Render restarts the container; a supervisor that quietly restarted one
child would leave the service reporting healthy while running a combination
nobody deployed.

**It makes no scientific decision and can make none.** It verifies the baked
serving bundle by running the project's own verifier in a subprocess, and if
that fails the container exits rather than starting a degraded service. It reads
no model, names no strategy, sets no threshold, and passes no scientific
environment variable to any child -- the children read the image's own ENV,
which is fixed at build time.

STDLIB ONLY, AND ON PURPOSE

This process lives for the lifetime of the container. Importing the project
package here would pull numpy, scikit-learn and pandas into PID 1 and hold their
memory for the whole deployment, on a 512 MiB budget, for a process whose entire
job is to wait. The bundle verification runs as a subprocess that exits and
gives its memory back.

NO BUSY LOOP

Signals are delivered to a self-pipe (:func:`signal.set_wakeup_fd`) and the
supervisor blocks in :func:`select.select` on it. Once startup is finished the
select has no timeout at all: this process consumes no CPU whatsoever until a
child dies or Render sends SIGTERM, which on a 0.1 CPU instance is the
difference between a demo that responds and one that does not.

The two startup waits do poll -- an HTTP readiness probe has to -- but they poll
by *sleeping inside the same select*, so a SIGTERM arriving during a slow cold
start is acted on immediately instead of after the current sleep.

EVERY CHILD IS NAMED BY ABSOLUTE PATH

No child is resolved through ``PATH``. The first real deployment of this image
failed with ``proxy exited with code 127``, and 127 was this file's own
``os._exit`` in the forked child: the proxy -- alone among the three -- was
spawned as the bare name ``caddy`` via ``execvp``, so *any* child-side failure
(the file missing, the file not executable, the wrong architecture, an
exhausted address space) arrived as the same opaque number with no reason
attached. Both halves of that are fixed here. Children are exec'd by absolute
path with :func:`os.execv`, and a child that cannot exec says why on stderr
before it exits.

The proxy executable is additionally probed *before anything is forked*, so an
image that cannot serve says so in a second rather than after a ninety-second
cold start that was never going to end in a listening port.
"""

from __future__ import annotations

import contextlib
import errno
import os
import select
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NoReturn

# ---------------------------------------------------------------------------
# Fixed internal topology.
#
# These are loopback addresses inside one container's network namespace, not
# deployment settings: nothing outside this container can reach either port, and
# a variable that could move them would only be a way to point the proxy at
# something the container is not running.
# ---------------------------------------------------------------------------
API_HOST: Final[str] = "127.0.0.1"
API_PORT: Final[int] = 8000
DASHBOARD_HOST: Final[str] = "127.0.0.1"
DASHBOARD_PORT: Final[int] = 8501

API_HEALTH_URL: Final[str] = f"http://{API_HOST}:{API_PORT}/health"
DASHBOARD_HEALTH_URL: Final[str] = (
    f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}/_stcore/health"
)

#: Where the image bakes the prepared scientific state, and where the Caddyfile
#: and the console entry point live.
STATE_ROOT: Final[str] = "/srv/state"
CADDYFILE: Final[str] = "/etc/caddy/Caddyfile"
VERIFIER: Final[str] = "/app/scripts/verify_serving_bundle.py"

#: The proxy executable, named by absolute path and never looked up on ``PATH``.
#:
#: This is one half of a contract with ``Dockerfile.render``, which copies the
#: pinned Caddy binary to exactly this location and proves at build time that it
#: is there, is mode 0755, and runs;
#: ``tests/unit/deployment/test_render_contract.py`` asserts the two files agree
#: on the string, so the image and the supervisor cannot drift apart silently.
#:
#: A ``PATH`` lookup would make the deployment depend on an environment variable
#: that the platform, the base image, or an operator with a dashboard can all
#: change, to locate a file whose position this repository chose.
PROXY_BINARY: Final[str] = "/usr/local/bin/caddy"

#: How long a child may take to start serving. Generous because the target is a
#: 0.1 CPU instance doing a cold import of the scientific stack: on a laptop the
#: API is answering in a few seconds, and a ceiling tuned to a laptop would turn
#: a slow-but-working cold start into a crash loop.
API_STARTUP_TIMEOUT: Final[float] = 300.0
DASHBOARD_STARTUP_TIMEOUT: Final[float] = 180.0
PROXY_STARTUP_TIMEOUT: Final[float] = 60.0

#: How often a startup wait re-probes, and how long a child gets to shut down
#: cleanly before it is killed. The API's own graceful-shutdown budget is 20s,
#: so the grace period here is longer than that on purpose: killing at exactly
#: its deadline would race it.
PROBE_INTERVAL: Final[float] = 0.5
SHUTDOWN_GRACE: Final[float] = 25.0

#: How long the proxy executable gets to answer ``version`` during the preflight.
#: Generous for a 0.1 CPU instance; the binary does nothing but print.
PROXY_PROBE_TIMEOUT: Final[float] = 30.0

#: Exit codes. Distinct so a Render log says which invariant broke.
EXIT_OK: Final[int] = 0
EXIT_STARTUP_FAILED: Final[int] = 1
EXIT_BAD_PORT: Final[int] = 2
EXIT_BUNDLE_UNVERIFIED: Final[int] = 3
EXIT_CHILD_DIED: Final[int] = 4
EXIT_PROXY_UNAVAILABLE: Final[int] = 5

#: What a forked child exits with when it could not exec at all, as distinct
#: from a program that ran and then failed. 126 is the shell's long-standing
#: code for "found, but could not be executed", and using it here keeps the
#: previous meaning of 127 -- an opaque catch-all -- out of the logs entirely.
CHILD_EXEC_FAILED: Final[int] = 126


def log(message: str) -> None:
    """Write one line to stdout, unbuffered, which is what Render collects."""
    print(f"[entrypoint] {message}", flush=True)


def _sanitized_reason(failure: BaseException) -> str:
    """Return a short, safe description of an operating-system failure.

    ``strerror`` and the exception's class name, and deliberately nothing else.
    ``str(exc)`` on an :class:`OSError` interpolates the filename it was given,
    and the repr of a failure raised between ``fork`` and ``exec`` can carry
    fragments of the child's argument vector; neither belongs in a log line that
    exists only to say *which* precondition broke.

    Nothing produced here reaches an HTTP client. This process serves no
    requests -- it is PID 1 -- and the three that do serve requests never read
    its output.
    """
    strerror = getattr(failure, "strerror", None)
    if isinstance(strerror, str) and strerror:
        return strerror
    return type(failure).__name__


# ---------------------------------------------------------------------------
# The public port
# ---------------------------------------------------------------------------


def resolve_public_port(environ: dict[str, str] | None = None) -> int:
    """Return the port Render told this container to listen on.

    Render sets ``PORT``. It is read rather than assumed, and no value is
    hardcoded as a fallback: 10000 is merely what Render happens to use today,
    and a container that silently bound it when ``PORT`` was missing would pass
    every local test and then be unreachable the day that default changes.

    Raises:
        ValueError: when ``PORT`` is unset, non-numeric, or outside the range a
            TCP listener can take. Port 0 is refused too -- it means "any free
            port", which for a service whose whole contract is *this* port is a
            failure dressed as success.
    """
    source = os.environ if environ is None else environ
    raw = source.get("PORT")
    if raw is None or not raw.strip():
        raise ValueError(
            "PORT is not set. Render provides it to every web service; set it "
            "explicitly when running this image outside Render."
        )
    try:
        port = int(raw.strip())
    except ValueError:
        raise ValueError(f"PORT is not an integer (got {raw.strip()!r})") from None
    if not 1 <= port <= 65_535:
        raise ValueError(f"PORT must be between 1 and 65535 (got {port})")
    return port


# ---------------------------------------------------------------------------
# Children
# ---------------------------------------------------------------------------


class Child:
    """One supervised process, its own session leader.

    Each child gets its own session (``setsid``) so the whole tree it may spawn
    can be signalled as a process group. Docker delivers SIGTERM to PID 1 only;
    with sessions of their own, the children hear about shutdown exactly once,
    from here, in the order this file chooses.
    """

    def __init__(self, name: str, argv: Sequence[str]) -> None:
        self.name = name
        self.argv = list(argv)
        self.pid: int | None = None
        self.status: int | None = None

    def start(self) -> None:
        """Fork, put the child in a new session, and exec into *argv*.

        :func:`os.execv`, never :func:`os.execvp`. ``argv[0]`` is an absolute
        path this repository chose -- the interpreter running this file, or the
        proxy binary the image copied to a reviewed location -- so there is
        nothing for a ``PATH`` lookup to add except a way for the deployment to
        resolve onto a different file than the one that was reviewed.

        A child that cannot exec writes one sanitized line to stderr and exits
        with :data:`CHILD_EXEC_FAILED`, so the supervisor's account of its death
        names a cause instead of a number.
        """
        if not self.argv[0].startswith("/"):  # pragma: no cover - defensive
            raise ValueError(
                f"{self.name} must be started by absolute path, not {self.argv[0]!r}"
            )
        pid = os.fork()
        if pid == 0:  # pragma: no cover - the child never returns
            try:
                os.setsid()
                for number in (
                    signal.SIGTERM,
                    signal.SIGINT,
                    signal.SIGCHLD,
                    signal.SIGHUP,
                ):
                    signal.signal(number, signal.SIG_DFL)
                os.execv(self.argv[0], self.argv)
            except BaseException as failure:
                # os.write rather than print: this runs after fork and before
                # exec, where the inherited stdio buffers belong to a process
                # that is about to be replaced. Suppressed because a child that
                # cannot even report why it is failing must still exit with the
                # code that says it failed.
                with contextlib.suppress(OSError):
                    os.write(
                        2,
                        f"[entrypoint] cannot execute {self.name} "
                        f"({self.argv[0]}): {_sanitized_reason(failure)}\n".encode(),
                    )
                os._exit(CHILD_EXEC_FAILED)
        self.pid = pid
        log(f"started {self.name} (pid {pid})")

    @property
    def running(self) -> bool:
        """Whether this child has been started and has not been reaped."""
        return self.pid is not None and self.status is None

    def signal_group(self, number: int) -> None:
        """Send *number* to this child's whole process group, if it is alive."""
        if not self.running or self.pid is None:
            return
        try:
            os.killpg(self.pid, number)
        except ProcessLookupError:
            pass
        except OSError as exc:  # pragma: no cover - defensive
            if exc.errno != errno.ESRCH:
                raise

    def describe_exit(self) -> str:
        """Return a human-readable account of how this child ended."""
        if self.status is None:
            return "still running"
        if os.WIFSIGNALED(self.status):
            return f"killed by signal {os.WTERMSIG(self.status)}"
        code = os.WEXITSTATUS(self.status)
        if code == CHILD_EXEC_FAILED:
            return "could not be executed at all (the reason is on the line above)"
        return f"exited with code {code}"


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------


class Supervisor:
    """Start the three processes, then wait for a signal or a death."""

    def __init__(self) -> None:
        self.children: list[Child] = []
        self.terminating = False
        self._read_fd, self._write_fd = os.pipe()
        os.set_blocking(self._write_fd, False)
        os.set_blocking(self._read_fd, False)
        # A Python-level handler is what makes CPython write the signal number
        # to the wakeup pipe. SIG_IGN would not, and SIG_DFL for SIGTERM would
        # end this process outright, leaving three children behind. SIGCHLD's
        # handler does nothing beyond that write -- reaping happens in the
        # supervisor loop, not in a handler, where a re-entrant waitpid would
        # race the loop's own.
        signal.signal(signal.SIGCHLD, lambda *_: None)
        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, self._request_stop)
        signal.set_wakeup_fd(self._write_fd)

    def _request_stop(self, *_: object) -> None:
        """Record that this container has been asked to stop.

        Set from a signal handler and read from the main loop, which is the only
        safe division of labour here: everything that actually tears the
        deployment down runs outside handler context.
        """
        self.terminating = True

    # -- waiting ----------------------------------------------------------

    def _drain(self) -> None:
        """Discard whatever the signal handlers wrote to the pipe."""
        try:
            while os.read(self._read_fd, 4096):
                pass
        except BlockingIOError:
            pass
        except OSError as exc:  # pragma: no cover - defensive
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise

    def wait_for_event(self, timeout: float | None) -> None:
        """Block until a signal arrives or *timeout* elapses.

        ``timeout=None`` blocks forever, which is the steady state: this process
        is scheduled exactly when something happens to it and never otherwise.
        """
        try:
            ready, _, _ = select.select([self._read_fd], [], [], timeout)
        except InterruptedError:  # pragma: no cover - retried by the caller
            return
        if ready:
            self._drain()

    def reap(self) -> list[Child]:
        """Reap every finished child and return the supervised ones among them.

        ``waitpid(-1)`` is used rather than one call per child because PID 1
        inherits orphaned grandchildren: reaping only known pids would leave
        zombies in a container that never restarts. Unknown pids are reaped and
        dropped.
        """
        finished: list[Child] = []
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
            for child in self.children:
                if child.pid == pid and child.status is None:
                    child.status = status
                    finished.append(child)
                    break
        return finished

    # -- readiness --------------------------------------------------------

    def _probe(self, url: str) -> bool:
        """Return whether *url* answers 200 right now.

        A loopback URL this file composed from its own constants, never one that
        came from the environment or a request, which is why the scheme cannot
        be anything but ``http``.
        """
        try:
            with urllib.request.urlopen(url, timeout=3) as answer:
                return bool(answer.status == 200)
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def await_ready(self, child: Child, url: str, timeout: float) -> None:
        """Wait until *child* answers *url*, or raise.

        Raises:
            _StopRequestedError: when a stop was requested mid-startup. A cold start on
                a 0.1 CPU instance is slow enough that a deploy cancelled part
                way through is an ordinary event, not an error.
            RuntimeError: if the child dies while starting, or has not started
                serving within *timeout*. Either way the caller tears the
                deployment down; a proxy in front of a process that never came
                up would publish a port that answers nothing.
        """
        deadline = time.monotonic() + timeout
        while True:
            if self.terminating:
                raise _StopRequestedError
            if self._probe(url):
                log(f"{child.name} is serving")
                return
            self.wait_for_event(PROBE_INTERVAL)
            for finished in self.reap():
                raise RuntimeError(
                    f"{finished.name} {finished.describe_exit()} during startup"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"{child.name} did not start serving within {timeout:.0f}s"
                )

    # -- lifecycle --------------------------------------------------------

    def spawn(self, name: str, argv: Sequence[str]) -> Child:
        """Start a supervised child and remember it."""
        child = Child(name, argv)
        self.children.append(child)
        child.start()
        return child

    def shutdown(self) -> None:
        """Stop every child, newest first, and reap them.

        Reverse order is the whole point: the proxy stops accepting before the
        console it fronts goes away, and the console goes away before the API it
        reads from does. A stop that took the API down first would spend its
        last seconds rendering errors into somebody's browser.
        """
        for child in reversed(self.children):
            if child.running:
                log(f"stopping {child.name}")
                child.signal_group(signal.SIGTERM)

        deadline = time.monotonic() + SHUTDOWN_GRACE
        while any(child.running for child in self.children):
            if time.monotonic() > deadline:
                break
            self.wait_for_event(0.2)
            self.reap()

        for child in reversed(self.children):
            if child.running:
                log(f"{child.name} did not stop in time; killing it")
                child.signal_group(signal.SIGKILL)
        # A short bounded drain: SIGKILL is immediate, and the loop exists only
        # so the process table is clean before PID 1 exits.
        deadline = time.monotonic() + 5.0
        while any(child.running for child in self.children):
            if time.monotonic() > deadline:  # pragma: no cover - defensive
                break
            self.wait_for_event(0.1)
            self.reap()


class _StopRequestedError(Exception):
    """A stop was requested before startup finished."""


# ---------------------------------------------------------------------------
# Command lines
# ---------------------------------------------------------------------------


def api_command() -> list[str]:
    """Return the API command: loopback only, no reloader, graceful shutdown.

    ``--host 127.0.0.1`` is the security boundary this deployment rests on. One
    process in this container is reachable from outside it, and it is the proxy;
    binding the detection service to ``0.0.0.0`` would publish it on Render's
    public port the moment the proxy's routing policy had a gap.
    """
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "password_attack_detector.api.app:app",
        "--host",
        API_HOST,
        "--port",
        str(API_PORT),
        "--no-server-header",
        "--timeout-graceful-shutdown",
        "20",
    ]


def dashboard_command() -> list[str]:
    """Return the console command, bound to loopback for the same reason."""
    app = (
        "/app/.venv/lib/python3.12/site-packages/"
        "password_attack_detector/dashboard/app.py"
    )
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app,
        "--server.address",
        DASHBOARD_HOST,
        "--server.port",
        str(DASHBOARD_PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]


def proxy_command() -> list[str]:
    """Return the proxy command. The Caddyfile reads ``PORT`` from the environment.

    Named by absolute path for the same reason the other two are: the file this
    starts is the file ``Dockerfile.render`` put there, and not whichever
    ``caddy`` a ``PATH`` happened to resolve to.
    """
    return [PROXY_BINARY, "run", "--config", CADDYFILE, "--adapter", "caddyfile"]


def verify_bundle_command() -> list[str]:
    """Return the bundle verification command."""
    return [sys.executable, VERIFIER, "--state-root", STATE_ROOT]


# ---------------------------------------------------------------------------
# The proxy executable
# ---------------------------------------------------------------------------


def proxy_unavailable_reason(path: str = PROXY_BINARY) -> str | None:
    """Return why the proxy cannot be started, or ``None`` if it can.

    Four questions, in the order in which each subsumes the last, because "it
    did not start" is four different deployment faults wearing one exit code:

    1. **Is the file there at all?** A runtime image built without the ``COPY``
       that places the binary looks entirely healthy until the last of the
       three processes is started. This is the plainest of the four and not,
       on the evidence, the one the first deployment hit: an image built from
       the same tree on linux/amd64 carries a working proxy at this path. Which
       of the four it actually was is the question the old catch-all destroyed
       and this function exists to answer on the next deploy.
    2. **Is it a regular file?** A directory at that path is executable in the
       ``os.access`` sense and is not a program.
    3. **Is it executable by this account?** The image serves as an
       unprivileged user, and a mode that survived the build as root does not
       prove anything about the account that will actually spawn it.
    4. **Does it run?** The only one of the four that catches a binary lifted
       out of an Alpine image that turns out not to be self-contained: a
       dynamically-linked executable whose ELF interpreter is absent fails
       ``exec`` with ``ENOENT``, which is to say it reports itself missing while
       sitting right there with its executable bit set. ``version`` is used
       because it reads no configuration, opens no socket, and writes nothing.

    The returned string names the reviewed path and the operating system's own
    reason, and nothing else -- no environment, no ``PATH``, no account, no
    directory listing. It is written to the deployment log and never to a
    response.
    """
    binary = Path(path)
    if not binary.exists():
        return f"{path} does not exist in this image"
    if not binary.is_file():
        return f"{path} is not a regular file"
    if not os.access(binary, os.X_OK):
        return f"{path} is not executable by the account this container runs as"
    try:
        probe = subprocess.run(
            [path, "version"],
            capture_output=True,
            check=False,
            timeout=PROXY_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"{path} did not answer 'version' within {PROXY_PROBE_TIMEOUT:.0f}s"
    except OSError as failure:
        return f"{path} could not be executed: {_sanitized_reason(failure)}"
    if probe.returncode != 0:
        return f"{path} exited with code {probe.returncode} when asked its version"
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def verify_bundle() -> int:
    """Run the bundle verifier in a subprocess and return its exit code.

    A subprocess rather than an import: the verifier pulls in the project
    package, and this process is going to sit in a select for the lifetime of
    the container. Whatever the verification costs is returned to the operating
    system before the API is started.
    """
    completed = subprocess.run(verify_bundle_command(), check=False)
    return completed.returncode


def run() -> int:
    """Start the deployment and supervise it. Returns the process exit code."""
    try:
        port = resolve_public_port()
    except ValueError as failure:
        log(f"refusing to start: {failure}")
        return EXIT_BAD_PORT
    log(f"public port {port}")

    # Before the bundle, and long before anything is forked.
    #
    # The proxy is started *last*, because the public port must not exist until
    # both processes behind it are serving -- that ordering is what makes
    # Render's health check meaningful and it is preserved exactly. But the
    # question "can this image start a proxy at all" is answered by four
    # filesystem checks and one 20-millisecond subprocess, and there is no
    # reason to spend a ninety-second cold start discovering the answer is no.
    #
    # This is what the first deployment of this image needed and did not have:
    # it started an API, started a console, and only then found out that the
    # thing which owns the public port could not be executed.
    unavailable = proxy_unavailable_reason()
    if unavailable is not None:
        log(
            f"refusing to start: the proxy cannot be run ({unavailable}). "
            "Nothing would listen on the public port, so this image is not "
            "serviceable; rebuild it rather than restarting it."
        )
        return EXIT_PROXY_UNAVAILABLE
    log("proxy executable verified")

    log("verifying the baked serving bundle")
    if verify_bundle() != 0:
        log(
            "refusing to start: the baked serving bundle did not verify. This "
            "image is not serviceable; rebuild it rather than restarting it."
        )
        return EXIT_BUNDLE_UNVERIFIED

    # Constructed only after the two things that can refuse to start have both
    # passed, so a container that is not going to serve never installs signal
    # handlers or forks anything.
    supervisor = Supervisor()

    try:
        api = supervisor.spawn("api", api_command())
        supervisor.await_ready(api, API_HEALTH_URL, API_STARTUP_TIMEOUT)

        dashboard = supervisor.spawn("dashboard", dashboard_command())
        supervisor.await_ready(
            dashboard, DASHBOARD_HEALTH_URL, DASHBOARD_STARTUP_TIMEOUT
        )

        # The proxy starts last, so the public port does not exist until both
        # processes behind it are serving. That ordering is what makes Render's
        # health check meaningful: it cannot succeed against a half-started
        # container, because there is nothing listening to succeed against.
        proxy = supervisor.spawn("proxy", proxy_command())
        supervisor.await_ready(
            proxy, f"http://127.0.0.1:{port}/healthz", PROXY_STARTUP_TIMEOUT
        )
    except _StopRequestedError:
        log("stop requested during startup")
        supervisor.shutdown()
        return EXIT_OK
    except RuntimeError as failure:
        log(f"startup failed: {failure}")
        supervisor.shutdown()
        return EXIT_STARTUP_FAILED

    log(f"serving on 0.0.0.0:{port}")

    while True:
        supervisor.wait_for_event(None)
        finished = supervisor.reap()
        if finished:
            for child in finished:
                log(f"{child.name} {child.describe_exit()}")
            log("a supervised process ended; stopping the container")
            supervisor.shutdown()
            return EXIT_CHILD_DIED
        if supervisor.terminating:
            log("stop requested; shutting down")
            supervisor.shutdown()
            return EXIT_OK


def main() -> NoReturn:
    """Run the supervisor and exit with its code."""
    sys.exit(run())


if __name__ == "__main__":
    main()
