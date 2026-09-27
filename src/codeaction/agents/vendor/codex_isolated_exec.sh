#!/usr/bin/env bash
# Run Codex with subscription auth while isolating every context-discovery path it has.
#
# Codex differs from the first vendor seat in one structural way: it does not read a token from a
# file, it reads a whole CODEX_HOME, and it WRITES there at startup (sqlite state, an installation
# id, a skills directory). The mounted account directory is therefore read-only and never used
# directly -- this script copies the one credential file out of it into a private throwaway home
# and points Codex at that. A run cannot mutate, exhaust, or cross-contaminate the account it
# authenticates as, and two accounts can run concurrently without sharing a lock.
set -euo pipefail

AUTH_DIR="${CODEACTION_CODEX_AUTH_DIR:?CODEACTION_CODEX_AUTH_DIR must name the mounted account}"
if [[ ! -r "${AUTH_DIR}/auth.json" ]]; then
  echo "Codex auth.json is not readable under: ${AUTH_DIR}" >&2
  exit 2
fi
CONFIG_SRC="${CODEACTION_CODEX_CONFIG:-/usr/local/share/codeaction/codex_config.toml}"
if [[ ! -r "$CONFIG_SRC" ]]; then
  echo "Codex config is not readable: ${CONFIG_SRC}" >&2
  exit 2
fi

# An OpenAI API key would take precedence over the subscription login and silently bill a key
# instead of the account this seat declares -- the same precedence trap the first seat hit with
# ANTHROPIC_API_KEY.
unset OPENAI_API_KEY OPENAI_API_BASE OPENAI_BASE_URL CODEX_API_KEY

CODEX_EXEC="${CODEX_BIN:-$(command -v codex || true)}"
if [[ -z "$CODEX_EXEC" ]]; then
  echo "codex executable not found" >&2
  exit 2
fi

# Not under /tmp: Codex refuses to create its PATH helper binaries when CODEX_HOME sits in a
# temporary directory, and warns on every invocation. /run/codeaction-codex-home is a tmpfs the
# service declares, which is private, writable, and not /tmp.
ISO_ROOT="$(mktemp -d "${CODEACTION_CODEX_STATE_DIR:-/run/codeaction-codex-home}/session.XXXXXX")"
cleanup() {
  rm -rf "$ISO_ROOT"
}
trap cleanup EXIT INT TERM

mkdir -p "$ISO_ROOT/home" "$ISO_ROOT/codex" "$ISO_ROOT/cwd" "$ISO_ROOT/catalog-home"
chmod 700 "$ISO_ROOT" "$ISO_ROOT/home" "$ISO_ROOT/codex" "$ISO_ROOT/cwd" "$ISO_ROOT/catalog-home"
install -m 0600 "${AUTH_DIR}/auth.json" "$ISO_ROOT/codex/auth.json"
install -m 0400 "$CONFIG_SRC" "$ISO_ROOT/codex/config.toml"

export HOME="$ISO_ROOT/home"
export XDG_CONFIG_HOME="$ISO_ROOT/home/.config"
export XDG_CACHE_HOME="$ISO_ROOT/home/.cache"
export XDG_STATE_HOME="$ISO_ROOT/home/.state"
export CODEX_HOME="$ISO_ROOT/codex"
cd "$ISO_ROOT/cwd"

# Online catalog refresh can change developer instructions even with the CLI binary pinned.
# Use that binary's bundled catalog for both the prompt check and the actual model session.
CODEX_HOME="$ISO_ROOT/catalog-home" "$CODEX_EXEC" debug models --bundled > "$ISO_ROOT/models.json"
{
  printf 'model_catalog_json = "%s"\n' "$ISO_ROOT/models.json"
  cat "$CONFIG_SRC"
} > "$ISO_ROOT/config.toml"
install -m 0400 "$ISO_ROOT/config.toml" "$CODEX_HOME/config.toml"

# Fail closed BEFORE the first model call. codex_config.toml is a deny list of feature names and
# a deny list cannot name a feature that does not exist yet; this reads the CLI's own report of
# effective state and refuses anything enabled outside the allowlist. It also proves the home
# holds nothing but our two files -- $CODEX_HOME/AGENTS.md and $CODEX_HOME/skills/* are injected
# into the prompt on 0.154.0 regardless of every flag we set.
# The text-purity half renders the model-visible input for THE prompt this episode will send --
# the final argument -- and the model this episode declares (`-c model=...`), so what is checked
# is what will run, not a stand-in.
GATE_OUT="${CODEACTION_VENDOR_OUTPUT_DIR:-/run/codeaction-vendor}/vendor_feature_gate.json"
GATE_PROMPT="$ISO_ROOT/prompt.txt"
printf '%s' "${@: -1}" > "$GATE_PROMPT"
GATE_MODEL=""
for arg in "$@"; do
  case "$arg" in model=*) GATE_MODEL="${arg#model=}" ;; esac
done
python3 "${CODEACTION_CODEX_GATE:-/usr/local/bin/codex_feature_gate.py}" \
  --codex-home "$CODEX_HOME" \
  --allowlist "${CODEACTION_CODEX_FEATURE_ALLOWLIST:-/usr/local/share/codeaction/codex_feature_allowlist.txt}" \
  --preamble-manifest "${CODEACTION_CODEX_PREAMBLE_MANIFEST:-/usr/local/share/codeaction/codex_preamble_manifest.json}" \
  --model "${GATE_MODEL:?the codex invocation must carry -c model=<id>}" \
  --prompt-file "$GATE_PROMPT" \
  --model-catalog "$ISO_ROOT/models.json" \
  --scratch-root "$ISO_ROOT" \
  --output "$GATE_OUT" \
  --codex-bin "$CODEX_EXEC"

set +e
"$CODEX_EXEC" "$@"
rc=$?
set -e
# The rollout and thread state are the only trace of code-mode execution (the JSON stream shows
# MCP calls but not the JavaScript around them). Copy them out before the home is destroyed;
# auth.json is excluded by name so no credential leaves with the evidence.
STATE_OUT="${CODEACTION_VENDOR_OUTPUT_DIR:-/run/codeaction-vendor}/codex_home_state"
mkdir -p "$STATE_OUT"
if [[ -d "$CODEX_HOME/sessions" ]]; then cp -r "$CODEX_HOME/sessions" "$STATE_OUT/"; fi
for f in "$CODEX_HOME"/*.sqlite "$CODEX_HOME"/*.sqlite-wal; do
  [[ -e "$f" ]] && cp "$f" "$STATE_OUT/"
done
find "$STATE_OUT" -name auth.json -delete
exit "$rc"
