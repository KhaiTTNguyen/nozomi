#!/bin/bash
# Build + run the intra/extra diffusion simulations for the high-dispersion
# (myelinated) substrates in highdisp_20260727_140957 and highdisp_20260727_140958.
#
# `batch_simulation build-scan` SCANS the experiment output folders directly for
# converged substrates (a run folder with data/*.pkl), skips compartments that are
# already simulated (sim/ADCdata/diffcoeff_<compartment>_*.pkl present), and emits
# the remaining jobs: compartment "intra" (inner_fibers) and "extra"
# (outer_fibers). Myelin is detected automatically from each substrate pickle
# (g_ratio / is_myelinated) -- no myelin flag needed. Outputs are written per
# substrate to <substrate>/sim/ADCdata/diffcoeff_<compartment>_*.pkl.
#
# The run is manifest-state-driven and fully resumable. By default it is launched
# DETACHED (setsid + nohup) so it survives the terminal/SSH session ending; a run
# log is written under experiment/setup/simulation/batch/logs/ and the PID +
# monitor commands are printed. Pass --foreground to run in the current shell.
#
# Usage:
#   # Scan both folders + run detached on GPUs 6,7 (skips already-simulated):
#   ./run-scripts/run-highdisp-simulation.sh
#
#   # Explicit scan dirs / diffusion time:
#   ./run-scripts/run-highdisp-simulation.sh \
#       --substrate-dirs "experiment/aim2-prep/data-high-dispersion/highdisp_20260727_140957 experiment/aim2-prep/data-high-dispersion/highdisp_20260727_140958" \
#       --sim-time 100
#
#   # RESUME an interrupted run (reuse an existing sim manifest, skip the build):
#   ./run-scripts/run-highdisp-simulation.sh --manifest <sim_manifest.json> --recover
#
#   # Re-simulate everything, including substrates that already have sim output:
#   ./run-scripts/run-highdisp-simulation.sh --include-existing-sim
#
#   # Preview what would run (foreground); trial only the first N jobs:
#   ./run-scripts/run-highdisp-simulation.sh --dry-run
#   ./run-scripts/run-highdisp-simulation.sh --limit 4
#
#   # Override GPUs:
#   ./run-scripts/run-highdisp-simulation.sh --gpus 6,7
#
# Any extra flags (--recover, --retry-failed, --limit, --dry-run) are forwarded to
# `batch_simulation run`.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python3}"

GPUS="6,7"
SUBSTRATE_DIRS=(
    "experiment/aim2-prep/data-high-dispersion/highdisp_20260727_140957"
    "experiment/aim2-prep/data-high-dispersion/highdisp_20260727_140958"
)
SIM_CONFIG="experiment/setup/simulation/default-sim.json"
SIM_TIME="100"
MANIFEST=""
INCLUDE_EXISTING=0
FOREGROUND=0
PASSTHROUGH=()

# Parse args: pull out flags we handle, forward the rest to `batch_simulation run`.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)                  GPUS="$2"; shift 2 ;;
        --gpus=*)                GPUS="${1#*=}"; shift ;;
        --manifest)              MANIFEST="$2"; shift 2 ;;
        --manifest=*)            MANIFEST="${1#*=}"; shift ;;
        --substrate-dirs)        IFS=' ' read -r -a SUBSTRATE_DIRS <<< "$2"; shift 2 ;;
        --substrate-dirs=*)      IFS=' ' read -r -a SUBSTRATE_DIRS <<< "${1#*=}"; shift ;;
        --sim-config)            SIM_CONFIG="$2"; shift 2 ;;
        --sim-config=*)          SIM_CONFIG="${1#*=}"; shift ;;
        --sim-time)              SIM_TIME="$2"; shift 2 ;;
        --sim-time=*)            SIM_TIME="${1#*=}"; shift ;;
        --include-existing-sim)  INCLUDE_EXISTING=1; shift ;;
        --foreground)            FOREGROUND=1; shift ;;
        --dry-run)               FOREGROUND=1; PASSTHROUGH+=("$1"); shift ;;  # dry-run stays attached
        *)                       PASSTHROUGH+=("$1"); shift ;;
    esac
done

# Build a fresh sim manifest unless one was supplied (supplying --manifest is the
# resume path and must NOT rebuild/overwrite the existing manifest).
if [[ -z "$MANIFEST" ]]; then
    echo "Scanning for converged substrates and building sim manifest..."
    BUILD_ARGS=(-m simulation_toolkit.cli.batch_simulation build-scan
        --substrate-dirs "${SUBSTRATE_DIRS[@]}"
        --sim-config "$SIM_CONFIG"
        --sim-time "$SIM_TIME")
    if [[ "$INCLUDE_EXISTING" -eq 1 ]]; then
        BUILD_ARGS+=(--include-existing-sim)
    fi
    BUILD_OUT="$("$PY" "${BUILD_ARGS[@]}")"
    echo "$BUILD_OUT"
    MANIFEST="$(echo "$BUILD_OUT" | sed -n 's/^Wrote //p' | head -n1)"
    if [[ -z "$MANIFEST" ]]; then
        echo "ERROR: could not determine sim manifest path from builder output." >&2
        exit 1
    fi
fi

echo ""
echo "Sim manifest : $MANIFEST"
echo "GPUs         : $GPUS"
echo ""

RUN_ARGS=(-m simulation_toolkit.cli.batch_simulation run
    --manifest "$MANIFEST"
    --gpus "$GPUS"
    "${PASSTHROUGH[@]}")

if [[ "$FOREGROUND" -eq 1 ]]; then
    exec "$PY" "${RUN_ARGS[@]}"
fi

# Detached launch: survive terminal/SSH disconnect. Log to a timestamped file.
LOG_DIR="experiment/setup/simulation/batch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/run_${STAMP}.log"

setsid nohup "$PY" "${RUN_ARGS[@]}" >"$RUN_LOG" 2>&1 < /dev/null &
RUN_PID=$!

echo "Launched DETACHED (survives disconnect)."
echo "  PID     : $RUN_PID"
echo "  Run log : $RUN_LOG"
echo ""
echo "Monitor:"
echo "  tail -f $RUN_LOG"
echo "  $PY -m simulation_toolkit.cli.batch_simulation status --manifest $MANIFEST"
echo "Stop:"
echo "  kill $RUN_PID   # then optionally: $PY -m simulation_toolkit.cli.batch_simulation reset --manifest $MANIFEST"
