#!/usr/bin/env bash
# Render ep06_level11 at 1080p: picture + score (+ voice if episodes/ep06_level11/voice/wav/ exists) -> out/ep06_level11/
# Extra args pass through, e.g. --jobs 16, --start 30 --end 60, --crf 16
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m backrooms.render ep06_level11 --all --res 1080 "$@"
