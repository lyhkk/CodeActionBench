#!/usr/bin/env bash
# Codex's remaining native tools are bounded by the container's mount table, not by argv. This
# entry point handles credential precedence and delegates CODEX_HOME isolation to the audited
# helper, then captures the JSONL stream and the process exit code the way the first vendor seat
# does -- same two files, same names, so the controller reads one shape for every seat.
set -euo pipefail

: "${CODEACTION_CODEX_AUTH_DIR:=/run/secrets/codeaction/codex-home}"
export CODEACTION_CODEX_AUTH_DIR
unset OPENAI_API_KEY OPENAI_API_BASE OPENAI_BASE_URL CODEX_API_KEY
: "${CODEACTION_VENDOR_OUTPUT_DIR:=/run/codeaction-vendor}"
mkdir -p "$CODEACTION_VENDOR_OUTPUT_DIR"

set +e
/usr/local/bin/codex_isolated_exec.sh "$@" </dev/null \
  | tee "$CODEACTION_VENDOR_OUTPUT_DIR/vendor_stream.jsonl"
rc=${PIPESTATUS[0]}
set -e
printf '{"cli_exit_code":%d}\n' "$rc" \
  >"$CODEACTION_VENDOR_OUTPUT_DIR/vendor_cli_exit.json"
exit "$rc"
