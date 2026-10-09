#!/usr/bin/env bash
# Render ep05_level10 at 1080p: picture + score (+ voice if episodes/ep05_level10/voice/wav/ exists) -> out/ep05_level10/
# Extra args pass through, e.g. --jobs 16, --start 30 --end 60, --crf 16
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m backrooms.render ep05_level10 --all --res 1080 "$@"
