#!/bin/bash
# Thin launcher for the NOZOMI batch Monte Carlo simulation orchestrator.
#
# Examples:
#   # Build a smoke sim manifest from a substrate manifest, sim_time=5 ms:
#   ./run-scripts/run-simulation-batch.sh build-smoke \
#       --substrate-manifest experiment/setup/substrate/batch/<smoke.json> \
#       --sim-time 5
#
#   # Build a full sim manifest, sim_time=100 ms:
#   ./run-scripts/run-simulation-batch.sh build \
#       --substrate-manifest experiment/setup/substrate/batch/<full.json> \
#       --sim-time 100
#
#   # Run on GPUs 0,1:
#   ./run-scripts/run-simulation-batch.sh run \
#       --manifest experiment/setup/simulation/batch/<file.json> \
#       --gpus 0,1
#
#   # Status / reset:
#   ./run-scripts/run-simulation-batch.sh status --manifest <file.json>
#   ./run-scripts/run-simulation-batch.sh reset  --manifest <file.json>
#
# This script does NOT pin a single GPU via CUDA_VISIBLE_DEVICES — the
# orchestrator manages per-process GPU pinning internally.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python3 -m simulation_toolkit.cli.batch_simulation "$@"
