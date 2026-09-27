#!/usr/bin/env bash
# Build the profile images, then run doctor and the free release checks.
#
# Usage: bash tools/rebuild_and_gate.sh [PROFILE] [TASK]
# Profile: reference-mcp | reference-code-first | vendor-mcp-direct | vendor-mcp-gateway
set -euo pipefail

PROFILE="${1:-reference-mcp}"
TASK="${2:-click_bell}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG_DIR="runs/_logs"
PYTHON="${CODEACTION_PYTHON:-python}"
mkdir -p "$LOG_DIR"

MODEL_MODE=()
RUN_SMOKE=1
RUN_VENDOR_HEALTH=0
case "$PROFILE" in
  reference-mcp|reference-code-first)
    AGENT_MODE=reference
    MODEL_MODE=(--reference-model-mode scripted)
    ;;
  vendor-mcp-direct)
    AGENT_MODE=fixture
    RUN_VENDOR_HEALTH=1
    ;;
  vendor-mcp-gateway)
    AGENT_MODE=claude
    RUN_SMOKE=0
    ;;
  *)
    echo "unknown interface profile: $PROFILE" >&2
    exit 2
    ;;
esac

bash tools/build_images.sh "$PROFILE"

COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
export PYTHONPATH="$PWD/src"
echo "running doctor"
"$PYTHON" -m codeaction.cli.main doctor \
  --agent-mode "$AGENT_MODE" \
  --interface-profile "$PROFILE" \
  "${MODEL_MODE[@]}" \
  --task "$TASK"

if [[ "$RUN_SMOKE" == "0" ]]; then
  echo "images aligned; smoke skipped because the profile needs a vendor subscription"
  exit 0
fi

echo "running credential-free smoke"
"$PYTHON" -m codeaction.cli.main smoke \
  --agent-mode "$AGENT_MODE" \
  --interface-profile "$PROFILE" \
  "${MODEL_MODE[@]}" \
  --task "$TASK" >"$LOG_DIR/gate_${PROFILE}.log" 2>&1 || {
    echo "smoke failed; see $LOG_DIR/gate_${PROFILE}.log" >&2
    tail -20 "$LOG_DIR/gate_${PROFILE}.log" >&2
    exit 1
  }

if [[ "$RUN_VENDOR_HEALTH" == "1" ]]; then
  echo "running credential-free real Claude MCP health"
  "$PYTHON" -m codeaction.cli.main mcp-health \
    --agent-mode claude \
    --interface-profile vendor-mcp-direct \
    --task "$TASK" >"$LOG_DIR/gate_vendor_mcp_health.log" 2>&1 || {
      echo "vendor MCP health failed; see $LOG_DIR/gate_vendor_mcp_health.log" >&2
      tail -20 "$LOG_DIR/gate_vendor_mcp_health.log" >&2
      exit 1
    }
  echo "CODEACTION VENDOR MCP HEALTH GREEN commit=$COMMIT"
fi

echo "CODEACTION GATE GREEN profile=$PROFILE commit=$COMMIT"
