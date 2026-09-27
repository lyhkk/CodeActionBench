#!/usr/bin/env bash
# Manage published dependency images using the repository-local host environment.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python="${CODEACTION_PYTHON:-$root/.codeaction-env/bin/python}"
cd "$root"
exec "$python" -I -m codeaction.image_distribution --root "$root" "$@"
