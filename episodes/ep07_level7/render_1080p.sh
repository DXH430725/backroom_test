#!/usr/bin/env bash
# Render ep07_level7 at 1080p: picture + score (+ voice if episodes/ep07_level7/voice/wav/ exists) -> out/ep07_level7/
# Extra args pass through, e.g. --jobs 16, --start 30 --end 60, --crf 16
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m backrooms.render ep07_level7 --all --res 1080 "$@"
