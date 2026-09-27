#!/usr/bin/env bash
# Install the host CLI, prepare credentials, verify assets and build dependency images.
set -euo pipefail
# These are child-shell changes only; do not inherit another checkout's runtime overrides.
for variable in ${!CODEACTION_@} ${!BENCH_@}; do unset "$variable"; done
unset PYTHONPATH PYTHONHOME
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
usage() {
  echo 'usage: bash tools/setup.sh {venv|conda} [--assets-root DIRECTORY] [--profile all|reference-mcp|vendor-mcp-direct] [--build | --images-manifest FILE] [--no-cache]'
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
kind=${1:-}
case "$kind" in venv|conda) shift ;; *) usage >&2; exit 2 ;; esac
assets="$HOME/.cache/codeaction/assets"
profile=all
cache=()
build=0
manifest="$root/docker/images.json"
explicit_manifest=0
while (( $# )); do
  case "$1" in
    --assets-root) assets=${2:?missing assets directory}; shift 2 ;;
    --profile) profile=${2:?missing profile}; shift 2 ;;
    --no-cache) cache=(--no-cache); shift ;;
    --build) build=1; shift ;;
    --images-manifest) manifest=${2:?missing image manifest}; explicit_manifest=1; shift 2 ;;
    *) usage >&2; exit 2 ;;
  esac
done
case "$profile" in all|reference-mcp|vendor-mcp-direct) ;; *) usage >&2; exit 2 ;; esac
if (( build && explicit_manifest )); then echo 'Choose --build or --images-manifest.' >&2; exit 2; fi
if (( explicit_manifest )) && [[ ! -f "$manifest" ]]; then echo 'Image manifest not found.' >&2; exit 2; fi
if (( ! build )) && [[ ! -f "$manifest" ]]; then
  echo 'This source checkout has no published image manifest; building its dependency images.'
  build=1
fi
if (( ! build && ${#cache[@]} )); then echo '--no-cache requires --build.' >&2; exit 2; fi
cd "$root"
for command in docker nvidia-smi curl unzip sha256sum shasum; do command -v "$command" >/dev/null; done
docker info >/dev/null
docker compose version
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
if [[ ! -x .codeaction-env/bin/python ]]; then
  if [[ "$kind" == venv ]]; then
    python3 -c 'import ensurepip' >/dev/null 2>&1 || {
      echo 'ensurepip unavailable. With Conda installed, run: bash tools/setup.sh conda' >&2
      exit 1
    }
    python3 -m venv .codeaction-env || {
      echo 'venv unavailable. With Conda installed, run: bash tools/setup.sh conda' >&2
      exit 1
    }
  else
    command -v conda >/dev/null || { echo 'Activate your Conda installation first; see QUICKSTART section 2.' >&2; exit 1; }
    conda create -y --prefix "$root/.codeaction-env" python=3.10 pip
  fi
fi
python="$root/.codeaction-env/bin/python"
# A failed venv can leave a Python link without pip; allow the documented Conda fallback.
if ! "$python" -I -m pip --version >/dev/null 2>&1; then
  if [[ "$kind" == conda ]]; then
    conda create -y --prefix "$root/.codeaction-env" python=3.10 pip
  else
    echo 'pip unavailable. Run: bash tools/setup.sh conda' >&2
    exit 1
  fi
fi
"$python" -I -m pip install -e .
if (( ! build )); then
  "$python" -I -m codeaction.image_distribution check "$manifest" --profile "$profile"
fi
umask 077
install -d -m 700 "$HOME/.config/codeaction"
for pair in secrets/provider.env.example:provider.env configs/provider-rate-limits.example.json:rate-limits.json configs/agents.example.json:agents.json; do
  source=${pair%%:*}
  target="$HOME/.config/codeaction/${pair#*:}"
  if [[ ! -e "$target" ]]; then cp "$source" "$target"; chmod 600 "$target"; fi
done
bash tools/download_assets.sh "$assets"
if (( build )); then
  CODEACTION_PYTHON="$python" bash tools/build_images.sh "$profile" "${cache[@]}"
  "$python" -I -m codeaction.image_distribution capture --profile "$profile"
else
  "$python" -I -m codeaction.image_distribution pull "$manifest" --profile "$profile"
fi
"$python" -I - "$assets" <<'PY'
import json, sys
from pathlib import Path
directory = Path('configs/local')
directory.mkdir(parents=True, exist_ok=True)
path = directory / 'reproduction.json'
private = Path.home() / '.config/codeaction'
path.write_text(json.dumps({
    'assets_root': str(Path(sys.argv[1]).expanduser().resolve()),
    'provider_env_file': str(private / 'provider.env'),
    'provider_rate_limit_file': str(private / 'rate-limits.json'),
}, indent=2) + '\n')
print('Setup complete. Resource and credential locations saved in configs/local/reproduction.json.')
print('Fill ~/.config/codeaction/provider.env and rate-limits.json for API models.')
print('For a vendor login: bash tools/reproduce.sh auth codex (or claude).')
PY
