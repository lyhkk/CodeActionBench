#!/usr/bin/env bash
# Replay fixed tool calls using only Docker, the simulator image and installed assets.
set -euo pipefail
TASK=${1:?usage: replay_container.sh TASK [OUTPUT_DIRECTORY]}
[[ "$TASK" =~ ^[A-Za-z0-9_]+$ ]] || { echo 'invalid task name' >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${2:-"$ROOT/runs/replay/$TASK"}
TASK_PACK=${CODEACTION_TASK_PACK_ROOT:-"$ROOT/benchmark/tasks"}
TASK_PACK=$(cd "$TASK_PACK" && pwd)
ASSETS=${CODEACTION_ASSETS_ROOT:-"$ROOT/assets"}
[[ -d "$ASSETS/objects" && -d "$ASSETS/embodiments" && -d "$ASSETS/background_texture" ]] || {
  echo 'set CODEACTION_ASSETS_ROOT to the installed resource tree' >&2; exit 2;
}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
ASSETS=$(cd "$ASSETS" && pwd)
exec docker run --rm --read-only --cap-drop ALL --security-opt no-new-privileges:true \
  --user "$(id -u):$(id -g)" --gpus "device=${CODEACTION_GPU:-0}" \
  --network none --tmpfs /tmp:rw,exec,size=2g \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache \
  -e PYTHONPATH=/opt/codeaction/src:/opt/codeaction \
  -v "${CODEACTION_SIM_CODE:-$ROOT}:/opt/codeaction:ro" \
  -v "${CODEACTION_SIM_CODE:-$ROOT}/backend/robotwin:/opt/robotwin:ro" \
  -e CODEACTION_RUN_CONTEXT=/opt/codeaction/config/context.json \
  -e CODEACTION_EXTENSIONS_FILE=/opt/codeaction/config/extensions.json \
  -v "$ASSETS:/opt/robotwin/assets:ro" -v "$OUT:/outputs" \
  -v "$TASK_PACK:/run/codeaction-task-pack:ro" \
  --entrypoint /Robotwin/conda/envs/robotwin/bin/python \
  "${CODEACTION_SIM_IMAGE:-codeaction-sim:dev}" \
  /opt/codeaction/oracle/replay.py "$TASK" --task-pack /run/codeaction-task-pack --out /outputs
