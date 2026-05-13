#!/bin/bash
# One-shot: confirm MOODS imports cleanly inside the apptainer container.
# Phase 3's motif scanning falls back to a (much more permissive) numpy
# scanner if MOODS is missing — so the cluster outputs would silently
# differ from what's expected. Run this on the cluster login node before
# submitting any Phase 3 job.
#
# Usage:
#   bash scripts/check_moods_in_container.sh

set -euo pipefail

CONTAINER="${CONTAINER:-container_0-1-3.sif}"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: container not found at $CONTAINER" >&2
    exit 1
fi

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
    --bind src/:/opt/app/src/ \
    "$CONTAINER" pixi run --manifest-path /opt/app/pixi.toml \
    python -c "
import MOODS.scan
import MOODS.tools
print('MOODS', MOODS.__version__ if hasattr(MOODS, '__version__') else 'imported')
print('MOODS.scan.Scanner:', MOODS.scan.Scanner)
print('MOODS.tools.threshold_from_p:', MOODS.tools.threshold_from_p)
print('OK -- Phase 3 motif scanning will use MOODS (exact thresholds)')
"
