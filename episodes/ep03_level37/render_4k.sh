#!/usr/bin/env bash
# Render ep03_level37 at 4k: picture + score (+ voice if episodes/ep03_level37/voice/wav/ exists) -> out/ep03_level37/
# Extra args pass through, e.g. --jobs 16, --start 30 --end 60, --crf 16
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m backrooms.render ep03_level37 --all --res 4k "$@"
