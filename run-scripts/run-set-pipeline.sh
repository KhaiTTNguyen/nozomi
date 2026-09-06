#!/bin/bash
# Set 1/2/3 study pipeline: add myelin (A), then effective-diameter stats (B) and
# intra/extra diffusion sims (C) IN PARALLEL. B and C only read the substrate
# pickles and write to disjoint folders (figs/ vs sim/), so they are safe to run
# together; both depend only on A. Phase A runs first and blocks; B and C are then
# launched DETACHED (setsid + nohup) with their own logs so they survive a
# terminal/SSH disconnect. Phase D (RDapp) and E (plots) are run separately after C.
#
# Launch the whole thing detached so even Phase A survives disconnect:
#   setsid nohup ./run-scripts/run-set-pipeline.sh --gpus 0,1,2,3 \
#       > experiment/aim2-prep/logs/pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#
# Usage:
#   ./run-scripts/run-set-pipeline.sh --gpus 0,1,2,3            # A, then B || C
#   ./run-scripts/run-set-pipeline.sh --gpus 0,1 --per-set      # tag sim jobs by set
#   ./run-scripts/run-set-pipeline.sh --gpus 0,1 --skip-add-inner   # resume: B || C only
#   ./run-scripts/run-set-pipeline.sh --gpus 0,1 --skip-diameter    # sims only
#
# Extra flags (--recover, --retry-failed, --limit, --dry-run) are forwarded to
# `batch_simulation run`.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Must be the sim_venv interpreter (system python3 fails to import NOZOMI modules).
PY="${PYTHON:-$REPO_ROOT/sim_venv/bin/python3}"

DATA_FULL="experiment/aim2-prep/data_full"
SIM_CONFIG="experiment/setup/simulation/default-sim.json"
SIM_TIME="100"
GPUS=""
PER_SET=0
SKIP_ADD_INNER=0
SKIP_DIAMETER=0
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)            GPUS="$2"; shift 2 ;;
        --gpus=*)          GPUS="${1#*=}"; shift ;;
        --sim-config)      SIM_CONFIG="$2"; shift 2 ;;
        --sim-config=*)    SIM_CONFIG="${1#*=}"; shift ;;
        --sim-time)        SIM_TIME="$2"; shift 2 ;;
        --sim-time=*)      SIM_TIME="${1#*=}"; shift ;;
        --per-set)         PER_SET=1; shift ;;
        --skip-add-inner)  SKIP_ADD_INNER=1; shift ;;
        --skip-diameter)   SKIP_DIAMETER=1; shift ;;
        *)                 PASSTHROUGH+=("$1"); shift ;;
    esac
done

if [[ -z "$GPUS" ]]; then
    echo "ERROR: --gpus is required (e.g. --gpus 0,1,2,3)." >&2
    exit 1
fi

LOG_DIR="experiment/aim2-prep/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

# ---- Phase A: add inner (myelin) compartment (blocking) --------------------
if [[ "$SKIP_ADD_INNER" -eq 0 ]]; then
    echo "[A] Adding inner (myelin) compartment to un-myelinated substrates..."
    "$PY" experiment/aim2-prep/add_inner_compartment.py
    echo "[A] done."
else
    echo "[A] skipped (--skip-add-inner)."
fi

# ---- Phase B: effective axon diameter stats (detached) ---------------------
if [[ "$SKIP_DIAMETER" -eq 0 ]]; then
    B_LOG="$LOG_DIR/phaseB_diameter_${STAMP}.log"
    setsid nohup "$PY" \
        simulation_toolkit/simulation_engine/helper/reprocess_diameter_stats.py \
        "$DATA_FULL" --skip-existing >"$B_LOG" 2>&1 < /dev/null &
    B_PID=$!
    echo "[B] diameter stats DETACHED  PID=$B_PID  log=$B_LOG"
else
    echo "[B] skipped (--skip-diameter)."
fi

# ---- Phase C: build sim manifest (scan) + run sims (detached) --------------
echo "[C] Building sim manifest (scan $DATA_FULL)..."
BUILD_ARGS=(-m simulation_toolkit.cli.batch_simulation build-scan
    --substrate-dirs "$DATA_FULL"
    --sim-config "$SIM_CONFIG"
    --sim-time "$SIM_TIME")
if [[ "$PER_SET" -eq 1 ]]; then
    BUILD_ARGS+=(--per-set)
fi
BUILD_OUT="$("$PY" "${BUILD_ARGS[@]}")"
echo "$BUILD_OUT"
MANIFEST="$(echo "$BUILD_OUT" | sed -n 's/^Wrote //p' | head -n1)"
if [[ -z "$MANIFEST" ]]; then
    echo "ERROR: could not determine sim manifest path from builder output." >&2
    exit 1
fi

C_LOG="$LOG_DIR/phaseC_sim_${STAMP}.log"
RUN_ARGS=(-m simulation_toolkit.cli.batch_simulation run
    --manifest "$MANIFEST" --gpus "$GPUS" "${PASSTHROUGH[@]}")
setsid nohup "$PY" "${RUN_ARGS[@]}" >"$C_LOG" 2>&1 < /dev/null &
C_PID=$!

echo ""
echo "[C] sims DETACHED  PID=$C_PID  log=$C_LOG"
echo "    manifest: $MANIFEST"
echo ""
echo "Monitor:"
[[ "$SKIP_DIAMETER" -eq 0 ]] && echo "  tail -f $B_LOG"
echo "  tail -f $C_LOG"
echo "  $PY -m simulation_toolkit.cli.batch_simulation status --manifest $MANIFEST"
echo ""
echo "After sims finish, run Phase D (RDapp) then E (plots):"
echo "  $PY simulation_toolkit/simulation_engine/helper/compute_rdapp_from_narrow_pulse.py $DATA_FULL"
echo "  $PY experiment/aim2-prep/plot_set_study.py --output-dir experiment/aim2-prep/plots/set2-study"
