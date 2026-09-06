#!/bin/bash
# Build + run the high-dispersion (K=2,4,7) substrate batch.
#
# Generates the missing Aim2 substrates and distributes generation across
# GPUs 3,4,5,6,7, writing output to experiment/aim2-prep/data-high-dispersion.
# Non-convergent / crashing jobs are flagged status="failed" by the orchestrator
# and the remaining jobs keep running.
#
# By default the run is launched DETACHED (setsid + nohup) so it survives the
# terminal/SSH session ending; a run log is written under
# experiment/setup/substrate/batch/logs/ and the PID + monitor commands are
# printed. Pass --foreground to run in the current shell instead.
#
# Usage:
#   ./run-scripts/run-highdisp-substrate.sh                 # build full manifest + run detached
#   ./run-scripts/run-highdisp-substrate.sh --smoke         # build 2-job 50-axon smoke + run detached
#   ./run-scripts/run-highdisp-substrate.sh --dry-run       # build + show what would run (foreground)
#   ./run-scripts/run-highdisp-substrate.sh --manifest <p>  # reuse an existing manifest
#   ./run-scripts/run-highdisp-substrate.sh --retry-failed  # re-run failed jobs
#   ./run-scripts/run-highdisp-substrate.sh --recover       # also pick up stale 'running' jobs
#   ./run-scripts/run-highdisp-substrate.sh --gpus 3,4      # override GPU list
#   ./run-scripts/run-highdisp-substrate.sh --foreground    # run attached to this shell
#
# Any extra flags are forwarded to `batch_substrate run`
# (e.g. --limit, --recover, --dry-run).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python3}"

GPUS="3,4,5,6,7"
OUTPUT_ROOT="experiment/aim2-prep/data-high-dispersion"
MANIFEST=""
SMOKE=0
FOREGROUND=0
PASSTHROUGH=()

# Parse args: pull out flags we handle, forward the rest to `batch_substrate run`.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)         GPUS="$2"; shift 2 ;;
        --gpus=*)       GPUS="${1#*=}"; shift ;;
        --manifest)     MANIFEST="$2"; shift 2 ;;
        --manifest=*)   MANIFEST="${1#*=}"; shift ;;
        --output-root)  OUTPUT_ROOT="$2"; shift 2 ;;
        --output-root=*) OUTPUT_ROOT="${1#*=}"; shift ;;
        --smoke)        SMOKE=1; shift ;;
        --foreground)   FOREGROUND=1; shift ;;
        --dry-run)      FOREGROUND=1; PASSTHROUGH+=("$1"); shift ;;  # dry-run stays attached
        *)              PASSTHROUGH+=("$1"); shift ;;
    esac
done

# Smoke overrides GPUs default to 6,7 unless the user set --gpus explicitly.
if [[ "$SMOKE" -eq 1 && "$GPUS" == "3,4,5,6,7" ]]; then
    GPUS="6,7"
fi

# Build a fresh manifest unless one was supplied.
if [[ -z "$MANIFEST" ]]; then
    if [[ "$SMOKE" -eq 1 ]]; then
        echo "Building high-dispersion SMOKE manifest..."
        BUILD_OUT="$("$PY" experiment/aim2-prep/build_highdisp_manifest.py --smoke)"
    else
        echo "Building high-dispersion manifest..."
        BUILD_OUT="$("$PY" experiment/aim2-prep/build_highdisp_manifest.py)"
    fi
    echo "$BUILD_OUT"
    MANIFEST="$(echo "$BUILD_OUT" | sed -n 's/^Wrote //p' | head -n1)"
    if [[ -z "$MANIFEST" ]]; then
        echo "ERROR: could not determine manifest path from builder output." >&2
        exit 1
    fi
fi

echo ""
echo "Manifest    : $MANIFEST"
echo "GPUs        : $GPUS"
echo "Output root : $OUTPUT_ROOT"
echo ""

RUN_ARGS=(-m simulation_toolkit.cli.batch_substrate run
    --manifest "$MANIFEST"
    --gpus "$GPUS"
    --output-root "$OUTPUT_ROOT"
    "${PASSTHROUGH[@]}")

if [[ "$FOREGROUND" -eq 1 ]]; then
    exec "$PY" "${RUN_ARGS[@]}"
fi

# Detached launch: survive terminal/SSH disconnect. Log to a timestamped file.
LOG_DIR="experiment/setup/substrate/batch/logs"
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
echo "  $PY -m simulation_toolkit.cli.batch_substrate status --manifest $MANIFEST"
echo "Stop:"
echo "  kill $RUN_PID   # then optionally: $PY -m simulation_toolkit.cli.batch_substrate reset --manifest $MANIFEST"
