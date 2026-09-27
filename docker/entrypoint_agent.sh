#!/usr/bin/env bash
# Claude's built-in filesystem/shell tools are restricted by argv from the controller. This entry
# point handles credential precedence and delegates HOME/cwd isolation to the audited helper.
set -euo pipefail

: "${CLAUDE_OAUTH_TOKEN_FILE:=/run/secrets/codeaction/claude_oauth_token.sh}"
export CLAUDE_OAUTH_TOKEN_FILE DISABLE_AUTOUPDATER=1
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
: "${CODEACTION_VENDOR_OUTPUT_DIR:=/run/codeaction-vendor}"
mkdir -p "$CODEACTION_VENDOR_OUTPUT_DIR"

set +e
/usr/local/bin/claude_isolated_exec.sh "$@" \
  | tee "$CODEACTION_VENDOR_OUTPUT_DIR/vendor_stream.jsonl"
rc=${PIPESTATUS[0]}
set -e
printf '{"cli_exit_code":%d}\n' "$rc" \
  >"$CODEACTION_VENDOR_OUTPUT_DIR/vendor_cli_exit.json"
exit "$rc"
