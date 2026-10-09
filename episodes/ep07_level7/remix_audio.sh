#!/usr/bin/env bash
# Rebuild the soundtrack (e.g. after voice wavs arrive) and re-mux every resolution already rendered.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m backrooms.render ep07_level7 --audio
for r in 1080 4k; do
  tag=$([ $r = 4k ] && echo 2160p || echo 1080p)
  if [ -f out/ep07_level7/ep07_level7_${tag}_picture.mp4 ]; then python3 -m backrooms.render ep07_level7 --mux --res $r; fi
done
