# Container images for the local demonstration deployment.
#
# One builder, one runtime base, and two runtime targets that differ only in the
# process they start. The scientific stack is installed once and shared, which
# is what makes the API container and the preparation container provably the
# same code -- a preparation step that trained under a different resolution of
# scikit-learn than the API loads would be publishing an artifact the API might
# refuse, and the refusal would arrive at startup rather than at build time.
#
# WHAT IS DELIBERATELY NOT HERE
#
# * No compiler, no build toolchain, and no package-manager cache in the runtime
#   layers. Every wheel this project needs publishes a manylinux build for
#   CPython 3.12 on x86_64, so nothing is compiled at image build time either.
# * No credential, token, key, or `.env`. The service accepts no credential
#   material, so a secret in an image layer would be one with nowhere to go.
# * No `.git`, no editor or local-tool state, no virtualenv from the host, no
#   tests, no test caches, no coverage output, no presentations. `.dockerignore`
#   is an allowlist rather than a handful of patterns, so anything that has not
#   been named on purpose is outside the build context -- including files that
#   did not exist when it was written.
# * No trained model, no artifact root, no serving bundle. Those are *outputs*
#   of a pipeline, not sources, and baking one into an image would make the
#   image the provenance of a scientific decision. The `prepare` service
#   produces them into a named volume before the API starts.
#
# Both runtime targets run as an unprivileged user, and neither writes anywhere
# in the image: compose gives them a read-only root filesystem with explicit
# tmpfs mounts for the few paths a framework insists on.

# ---------------------------------------------------------------------------
# Base -- pinned by digest as well as by tag.
#
# The tag says what it is to a human; the digest is what actually gets pulled,
# so a rebuilt `python:3.12-slim` upstream cannot silently change what this
# deployment was verified against. Bumping it is a deliberate edit with a
# rebuild behind it.
#
# python:3.12.13-slim -- Debian 13 (trixie), CPython 3.12.13
# ---------------------------------------------------------------------------
ARG PYTHON_IMAGE=python:3.12-slim@sha256:c3d81d25b3154142b0b42eb1e61300024426268edeb5b5a26dd7ddf64d9daf28

# ---------------------------------------------------------------------------
# Builder: resolve and install exactly what uv.lock names.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

# uv is pinned too. An installer that resolves differently between two builds
# would make "deterministic install from the lockfile" a claim about the
# lockfile only.
ARG UV_VERSION=0.12.5

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /app

# Dependencies first, and without the project, so a change to src/ does not
# re-resolve and re-download the scientific stack.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# --no-dev: pytest, mypy, ruff and pre-commit are development tools and have no
# business in a runtime image. --frozen: the lockfile is the input, and a build
# that updated it would be resolving rather than installing.
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# ---------------------------------------------------------------------------
# Runtime base: the virtual environment, the reviewed configurations, and the
# offline preparation script. No source tree, no tests, no toolchain.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    HOME=/home/pad \
    PAD_DEMO_STATE_ROOT=/srv/state

# A fixed uid/gid rather than whatever the distro assigns next: the named volume
# the preparation step writes and the API reads is owned by this account, and an
# account whose number moved between two builds would make that ownership a
# coincidence.
RUN groupadd --gid "${APP_GID}" pad \
 && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin pad

WORKDIR /app

COPY --from=builder --chown=root:root /app/.venv /app/.venv

# The reviewed configurations the deployment is defined by. Copied read-only and
# owned by root: the runtime account can read them and cannot edit them, so a
# compromised process cannot restate a scientific decision by rewriting the file
# that declares it.
COPY --chown=root:root configs ./configs
COPY --chown=root:root .streamlit ./.streamlit
COPY --chown=root:root scripts/prepare_demo_bundle.py ./scripts/prepare_demo_bundle.py

# Created here, owned by the runtime account, so an empty named volume mounted
# over it inherits that ownership. Without this the volume would arrive
# root-owned and the unprivileged preparation step could not write to it.
RUN install -d -o pad -g pad -m 0755 /srv/state

USER pad:pad

# ---------------------------------------------------------------------------
# API: the detection service, and the same image the preparation step runs in.
# ---------------------------------------------------------------------------
FROM runtime AS api

EXPOSE 8000

# Exec form, so uvicorn is PID 1 and receives SIGTERM directly. A shell-form
# command would put /bin/sh at PID 1, which does not forward signals -- the
# FastAPI lifespan shutdown would never run and `docker compose down` would end
# in a ten-second kill instead of a clean cancellation of the replay tasks.
CMD ["uvicorn", "password_attack_detector.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--no-server-header", "--timeout-graceful-shutdown", "20"]

# ---------------------------------------------------------------------------
# Dashboard: the analyst console. Holds no artifact and no scientific
# configuration -- compose gives it neither the state volume nor a PAD_API_*
# artifact path, so the only detection it can show is one the API performed.
# ---------------------------------------------------------------------------
FROM runtime AS dashboard

EXPOSE 8501

# The repository's .streamlit/config.toml binds 127.0.0.1, which is right for a
# laptop and wrong inside a network namespace nothing else can enter. The
# environment overrides the file for this deployment only; the tracked default
# stays loopback, so nobody gets a console on every interface by running it the
# ordinary way.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "/app/.venv/lib/python3.12/site-packages/password_attack_detector/dashboard/app.py"]
