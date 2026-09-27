#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python="$root/.codeaction-env/bin/python"
if [[ ! -x "$python" ]]; then
  echo 'Run bash tools/setup.sh venv (or conda) first.' >&2
  exit 1
fi
exec "$python" -I "$root/tools/reproduce.py" "$@"
