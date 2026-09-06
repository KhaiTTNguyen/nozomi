#!/bin/bash
# Run the wide_pulse validation tests and collect the plotted gradient waveforms
# under tests/validation/wide_pulse/test_figures/.
#
# GPU tests (test_01..test_05, test_gradient_sim_2boundary) require pycuda + a
# GPU; they are pinned to a single GPU via CUDA_VISIBLE_DEVICES. The non-GPU
# waveform-design tests (test_00, test_00b) render the achieved ideal-PGSE,
# slew-PGSE, trapezoidal-OGSE and apodized-OGSE protocol waveforms.
#
# By default the run is launched DETACHED (setsid + nohup) so it survives the
# terminal/SSH session ending; a timestamped log is written and the PID +
# monitor command are printed. Pass --foreground to run in the current shell.
#
# Usage:
#   ./run-scripts/run-wide-pulse-tests.sh                 # detached, GPU 6
#   ./run-scripts/run-wide-pulse-tests.sh --gpu 6
#   ./run-scripts/run-wide-pulse-tests.sh --foreground
#   ./run-scripts/run-wide-pulse-tests.sh -k test_00      # forward pytest args

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

GPU="6"
FOREGROUND=0
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)         GPU="$2"; shift 2 ;;
        --gpu=*)       GPU="${1#*=}"; shift ;;
        --foreground)  FOREGROUND=1; shift ;;
        *)             PASSTHROUGH+=("$1"); shift ;;
    esac
done

# Prefer the project venv if present.
if [[ -f "sim_venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source sim_venv/bin/activate
fi
PY="${PYTHON:-python3}"

export CUDA_VISIBLE_DEVICES="$GPU"

TEST_DIR="tests/validation/wide_pulse"
RUN_ARGS=(-m pytest "$TEST_DIR" -v "${PASSTHROUGH[@]}")

echo "Repo         : $REPO_ROOT"
echo "GPU          : $GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Figures out  : $TEST_DIR/test_figures/"
echo ""

if [[ "$FOREGROUND" -eq 1 ]]; then
    exec "$PY" "${RUN_ARGS[@]}"
fi

LOG_DIR="$TEST_DIR/test_figures/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/wide_pulse_tests_${STAMP}.log"

setsid nohup "$PY" "${RUN_ARGS[@]}" >"$RUN_LOG" 2>&1 < /dev/null &
RUN_PID=$!

echo "Launched DETACHED (survives disconnect)."
echo "  PID     : $RUN_PID"
echo "  Run log : $RUN_LOG"
echo ""
echo "Monitor:"
echo "  tail -f $RUN_LOG"
echo "Stop:"
echo "  kill $RUN_PID"
