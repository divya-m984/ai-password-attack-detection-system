#!/usr/bin/env bash
#
# Stop the containerized demonstration.
#
# The default stops and removes the containers and the project network, and
# keeps the serving-state volume -- so the next start skips the preparation
# pipeline and comes up in seconds. Pass --purge to discard the volume too; the
# next start then rebuilds the same scientific state from the same seeded
# configurations, which is what makes discarding it safe rather than expensive.
#
# This script touches no Git state and writes nothing into the repository.

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

purge=false
for argument in "$@"; do
    case "$argument" in
        --purge) purge=true ;;
        -h|--help)
            echo "usage: ${0##*/} [--purge]"
            echo
            echo "  (default)  stop the containers, keep the serving state"
            echo "  --purge    also discard the serving-state volume"
            exit 0
            ;;
        *)
            echo "unknown argument: $argument" >&2
            exit 2
            ;;
    esac
done

if ! docker info >/dev/null 2>&1; then
    echo "The Docker daemon is not reachable; nothing to stop." >&2
    exit 1
fi

if [ "$purge" = true ]; then
    echo "Stopping, and discarding the serving state."
    docker compose down --volumes --remove-orphans
else
    echo "Stopping. The serving state is kept; use --purge to discard it."
    docker compose down --remove-orphans
fi

echo
echo "Stopped. Nothing of this project is listening on 8000 or 8501."
