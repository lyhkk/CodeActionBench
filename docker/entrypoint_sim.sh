#!/usr/bin/env bash
# Entry for codeaction-sim: exec the MCP episode host. NO echo before exec —
# stdout (fd1) is the JSON-RPC channel and mcp_episode_server dups it first thing.
set -euo pipefail
: "${ROBOTWIN_ROOT:=/opt/robotwin}"
: "${PYOPENGL_PLATFORM:=egl}"
export ROBOTWIN_ROOT PYOPENGL_PLATFORM
exec /Robotwin/conda/envs/robotwin/bin/python \
  -m codeaction.runtime.sim_server "$@"
