#!/bin/sh
# Credential-free real-CLI MCP config/health gate. This never starts a model-backed session.
set -u

health_home=/tmp/codeaction-mcp-health
mkdir -p "$health_home"
output="$(
  env -i \
    HOME="$health_home" \
    PATH=/usr/local/bin:/usr/bin:/bin \
    TERM=dumb \
    LANG=C.UTF-8 \
    DISABLE_AUTOUPDATER=1 \
    CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 \
    CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 \
    CLAUDE_CODE_SKIP_PROMPT_HISTORY=1 \
    CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1 \
    CLAUDE_CODE_DISABLE_CRON=1 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    MCP_TIMEOUT=120000 \
    claude \
      --settings /usr/local/share/codeaction/mcp_health_settings.json \
      --mcp-config /run/codeaction/mcp.json \
      --strict-mcp-config \
      mcp list 2>&1
)"
rc=$?
printf '%s\n' "$output"

healthy=false
case "$output" in
  *codeaction*[Cc]onnected*) healthy=true ;;
esac
if [ "$rc" -eq 0 ] && [ "$healthy" = true ]; then
  printf '%s\n' \
    '[vendor-mcp-health] {"schema_version":"1.0","healthy":true,"server":"codeaction","strict_mcp_config":true,"credential_supplied":false}'
  exit 0
fi
printf '%s\n' \
  '[vendor-mcp-health] {"schema_version":"1.0","healthy":false,"server":"codeaction","strict_mcp_config":true,"credential_supplied":false}'
exit 2
