#!/usr/bin/env bash
# Pinned public resources; repeat the command to continue an interrupted download.
set -euo pipefail
if (( $# != 1 )); then
  echo 'usage: bash tools/download_assets.sh ASSETS_DIRECTORY' >&2
  exit 2
fi
assets=$1
revision=3dc3b798668feb99ac61cc9086d84cbcc3d79186
mkdir -p "$assets"
for spec in \
  embodiments:85ffeff55a5066def5931224a85cfa3f8abaa1fbf779bd17789a7fb3f85bc789 \
  objects:6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27 \
  background_texture:54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070; do
  name=${spec%%:*}
  expected=${spec#*:}
  if ! printf '%s  %s\n' "$expected" "$assets/$name.zip" | sha256sum -c - >/dev/null 2>&1; then
    curl -fL --retry 3 -C - \
      "https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/$revision/$name.zip" \
      -o "$assets/$name.zip"
  fi
  printf '%s  %s\n' "$expected" "$assets/$name.zip" | sha256sum -c -
  unzip -tq "$assets/$name.zip"
  if [[ ! -f "$assets/.$name.complete" || ! -d "$assets/$name" ]]; then
    unzip -q -o "$assets/$name.zip" -d "$assets"
    touch "$assets/.$name.complete"
  fi
  test -d "$assets/$name"
  test ! -L "$assets/$name"
done
test -f "$assets/embodiments/aloha-agilex/config.yml"
