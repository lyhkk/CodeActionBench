#!/usr/bin/env bash
# Build current source; Docker owns dependency resolution and layer caching.
# Usage: bash tools/build_images.sh [PROFILE] [--no-cache]
set -euo pipefail
PROFILE=${1:-reference-mcp}
CACHE_ARGS=()
case "${2:-}" in
  '') ;;
  --no-cache) CACHE_ARGS=(--no-cache) ;;
  *) echo "usage: bash tools/build_images.sh [PROFILE] [--no-cache]" >&2; exit 2 ;;
esac
if (( $# > 2 )); then
  echo "usage: bash tools/build_images.sh [PROFILE] [--no-cache]" >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON=${CODEACTION_PYTHON:-python}
case "$PROFILE" in
  all) IMAGES=(sim-base sim reference-agent fixture-agent gateway claude-agent codex-agent) ;;
  reference-mcp|reference-code-first) IMAGES=(sim-base sim reference-agent gateway) ;;
  vendor-mcp-direct) IMAGES=(sim-base sim fixture-agent gateway claude-agent codex-agent) ;;
  vendor-mcp-gateway) IMAGES=(sim-base sim claude-agent gateway scratch scratch-launcher) ;;
  *) echo "unknown interface profile: $PROFILE" >&2; exit 2 ;;
esac

CONTEXT_PARENT=$(mktemp -d)
trap 'rm -rf "$CONTEXT_PARENT"' EXIT
BUILD_CONTEXT="$CONTEXT_PARENT/source"
export PYTHONPATH="$PWD/src"
"$PYTHON" -m codeaction.release build-context "$PWD" "$BUILD_CONTEXT"

if [[ -e .git ]]; then
  COMMIT=$(git rev-parse HEAD)
  BUILD_DATE=$(git show -s --format=%cI HEAD)
else
  COMMIT=unknown
  BUILD_DATE=unknown
fi
mkdir -p runs/_logs
echo "Checking dependency environments (HEAD $COMMIT)"
for image in "${IMAGES[@]}"; do
  ENVIRONMENT_ID=$("$PYTHON" -c 'import sys; from pathlib import Path; from codeaction.environments import environment_identity; print(environment_identity(Path(sys.argv[1]), sys.argv[2]))' "$BUILD_CONTEXT" "$image")
  EXISTING=$(docker image inspect --format '{{ index .Config.Labels "org.codeaction.environment-sha256" }}' "codeaction-$image:dev" 2>/dev/null || true)
  if [[ ${#CACHE_ARGS[@]} == 0 && "$EXISTING" == "$ENVIRONMENT_ID" ]]; then
    echo "reused $image (dependencies unchanged)"
    continue
  fi
  extra=()
  # sim uses the base just built locally; pulling it would query a nonexistent registry.
  if [[ ${#CACHE_ARGS[@]} != 0 && "$image" != sim ]]; then
    extra+=(--pull)
  fi
  case "$image" in
    sim) extra+=(--build-arg "LOCK_SHA=$(shasum -a 256 "$BUILD_CONTEXT/docker/requirements.lock" | cut -d' ' -f1)") ;;
    reference-agent) extra+=(--build-arg "REFERENCE_LOCK_SHA=$(shasum -a 256 "$BUILD_CONTEXT/docker/reference-agent.requirements.lock" | cut -d' ' -f1)") ;;
  esac
  log="runs/_logs/build_${image}.log"
  # Keep the heavy simulator base independent of application source metadata.
  metadata=(--label "org.codeaction.environment-sha256=$ENVIRONMENT_ID")
  if [[ "$image" != sim-base ]]; then
    metadata+=(--build-arg "SOURCE_COMMIT=$COMMIT" --build-arg "BUILD_DATE=$BUILD_DATE")
  fi
  if docker build -f "$BUILD_CONTEXT/docker/$image.Dockerfile" -t "codeaction-$image:dev" \
      "${CACHE_ARGS[@]}" "${metadata[@]}" "${extra[@]}" "$BUILD_CONTEXT" >"$log" 2>&1; then
    echo "built $image (cache options: ${CACHE_ARGS[*]:-Docker defaults})"
  else
    echo "build failed: $image; see $log" >&2
    tail -20 "$log" >&2
    exit 1
  fi
done
