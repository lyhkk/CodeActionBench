#!/usr/bin/env bash
# Replay a published reference sequence on a simulation host.
# Usage: bash tools/run_oracle.sh replay <task> [replay.py args]
set -euo pipefail
MODE=${1:?usage: run_oracle.sh replay <task> [args]}
TASK=${2:?usage: run_oracle.sh replay <task> [args]}
shift 2
if [[ "$MODE" != "replay" ]]; then
  echo "unsupported mode $MODE: use replay" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYOPENGL_PLATFORM=egl
export PYTHONDONTWRITEBYTECODE=1
PY=${CODEACTION_PYTHON:-${PYTHON:-python}}
exec "$PY" oracle/replay.py "$TASK" "$@"
