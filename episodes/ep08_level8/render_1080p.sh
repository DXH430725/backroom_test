#!/usr/bin/env bash
# Render ep08_level8 at 1080p: picture + score (+ voice if episodes/ep08_level8/voice/wav/ exists) -> out/ep08_level8/
# Extra args pass through, e.g. --jobs 16, --start 30 --end 60, --crf 16
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m backrooms.render ep08_level8 --all --res 1080 "$@"
