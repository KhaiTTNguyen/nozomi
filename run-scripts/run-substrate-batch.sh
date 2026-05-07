#!/bin/bash
# Thin launcher for the NOZOMI batch substrate orchestrator.
#
# Examples:
#   ./run-scripts/run-substrate-batch.sh build
#   ./run-scripts/run-substrate-batch.sh build-smoke
#   ./run-scripts/run-substrate-batch.sh run --manifest experiment/setup/substrate/batch/<file>.json --gpus 0,1,2,3,4,5
#   ./run-scripts/run-substrate-batch.sh run --manifest <smoke.json> --gpus 0,1
#   ./run-scripts/run-substrate-batch.sh status --manifest <file.json>
#
# This script does NOT pin a single GPU via CUDA_VISIBLE_DEVICES — the
# orchestrator manages per-process GPU pinning internally.

set -e

# Move to repo root (parent of run-scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python3 -m simulation_toolkit.cli.batch_substrate "$@"
