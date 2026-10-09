#!/usr/bin/env bash
# Render ep01_level254 at 4k: picture + score (+ voice if episodes/ep01_level254/voice/wav/ exists) -> out/ep01_level254/
# Extra args pass through, e.g. --jobs 16, --start 30 --end 60, --crf 16
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m backrooms.render ep01_level254 --all --res 4k "$@"
