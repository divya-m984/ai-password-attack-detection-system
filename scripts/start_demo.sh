#!/usr/bin/env bash
#
# Start the containerized demonstration and print where to look.
#
# A thin wrapper, and thin on purpose: `docker compose up --build --wait` is the
# whole workflow, and everything below it is either a precondition worth failing
# on early or a URL worth printing at the end. There is no scientific logic here
# and there must not be -- what gets trained, which champion is frozen and which
# hybrid is selected are decided by the tracked configurations the `prepare`
# service reads, and a shell script that could influence any of that would be a
# place to change a result without leaving a record.
#
# This script touches no Git state and writes nothing into the repository.

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not on PATH." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "This needs Docker Compose v2 ('docker compose', not 'docker-compose')." >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "The Docker daemon is not reachable. Start it and try again." >&2
    exit 1
fi

echo "Building images and starting the demonstration."
echo
echo "The first run trains a champion before the API starts: the 'prepare'"
echo "service runs the real pipeline offline and takes about a minute. Nothing"
echo "is fitted at serving time, on this run or any later one."
echo

# --wait blocks until every long-running service reports healthy, and fails if
# one does not. Without it this script would print URLs that are not yet live.
docker compose up --build --wait

cat <<'URLS'

Running.

  API         http://localhost:8000
  Swagger     http://localhost:8000/docs
  Dashboard   http://localhost:8501

  Readiness   curl -s http://localhost:8000/ready
  Logs        docker compose logs -f api
  Stop        ./scripts/stop_demo.sh

Replay history lives in the API process's memory and is discarded when that
process restarts. The frozen scientific state is not: it is on a read-only
volume and survives until you remove it with `docker compose down -v`.
URLS
