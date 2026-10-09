#!/usr/bin/env bash
# Render ep04_level881 at 1080p: picture + score (+ voice if episodes/ep04_level881/voice/wav/ exists) -> out/ep04_level881/
# Extra args pass through, e.g. --jobs 16, --start 30 --end 60, --crf 16
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m backrooms.render ep04_level881 --all --res 1080 "$@"
