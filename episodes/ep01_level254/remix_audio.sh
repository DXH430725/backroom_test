#!/usr/bin/env bash
# Rebuild the soundtrack (e.g. after voice wavs arrive) and re-mux every resolution already rendered.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m backrooms.render ep01_level254 --audio
for r in 1080 4k; do
  tag=$([ $r = 4k ] && echo 2160p || echo 1080p)
  if [ -f out/ep01_level254/ep01_level254_${tag}_picture.mp4 ]; then python3 -m backrooms.render ep01_level254 --mux --res $r; fi
done
