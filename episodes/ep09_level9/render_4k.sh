#!/usr/bin/env bash
# Render ep09_level9 at 4k: picture + score (+ voice if episodes/ep09_level9/voice/wav/ exists) -> out/ep09_level9/
# Extra args pass through, e.g. --jobs 16, --start 30 --end 60, --crf 16
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m backrooms.render ep09_level9 --all --res 4k "$@"
