#!/usr/bin/env bash
# Run Claude Code with one declared credential and isolated project/user context discovery.
set -euo pipefail

# No host-path fallback: the caller supplies the location (compose mounts the token at a
# fixed in-container path). Guessing a filename under a home directory is what baked one
# operator's private path into every checkout.
TOKEN_FILE="${CLAUDE_OAUTH_TOKEN_FILE:?CLAUDE_OAUTH_TOKEN_FILE must name the token file}"
if [[ ! -r "$TOKEN_FILE" ]]; then
  echo "OAuth token file is not readable: ${TOKEN_FILE}" >&2
  exit 2
fi

AUTH_HELPER="${CODEACTION_CLAUDE_AUTH_HELPER:-$(dirname "${BASH_SOURCE[0]}")/claude_auth.py}"
AUTH_HELPER="$(cd "$(dirname "$AUTH_HELPER")" && pwd)/$(basename "$AUTH_HELPER")"
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL CLAUDE_CODE_OAUTH_TOKEN

CLAUDE_EXEC="${CLAUDE_BIN:-$(command -v claude || true)}"
if [[ -z "$CLAUDE_EXEC" ]]; then
  echo "claude executable not found" >&2
  exit 2
fi

ISO_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/codeaction-claude-isolated.XXXXXX")"
cleanup() {
  rm -rf "$ISO_ROOT"
}
trap cleanup EXIT INT TERM

mkdir -p "$ISO_ROOT/home" "$ISO_ROOT/cwd" "$ISO_ROOT/config" \
  "$ISO_ROOT/cache" "$ISO_ROOT/state"
chmod 700 "$ISO_ROOT" "$ISO_ROOT/home" "$ISO_ROOT/cwd" \
  "$ISO_ROOT/config" "$ISO_ROOT/cache" "$ISO_ROOT/state"

export HOME="$ISO_ROOT/home"
export XDG_CONFIG_HOME="$ISO_ROOT/config"
export XDG_CACHE_HOME="$ISO_ROOT/cache"
export XDG_STATE_HOME="$ISO_ROOT/state"
export CLAUDE_CONFIG_DIR="$ISO_ROOT/home/.claude"
cd "$ISO_ROOT/cwd"

python3 "$AUTH_HELPER" "$CLAUDE_EXEC" "$@"
