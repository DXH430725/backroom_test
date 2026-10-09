#!/usr/bin/env bash
# Render ep02_level0 at 4k: picture + score (+ voice if episodes/ep02_level0/voice/wav/ exists) -> out/ep02_level0/
# Extra args pass through, e.g. --jobs 16, --start 30 --end 60, --crf 16
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m backrooms.render ep02_level0 --all --res 4k "$@"
