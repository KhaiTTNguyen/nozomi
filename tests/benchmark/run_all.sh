#!/usr/bin/env bash
# Orchestrate the full benchmark: build substrate once, run each framework N times.
#
# Usage:
#   bash tests/benchmark/run_all.sh [framework ...]
#
# Examples:
#   bash tests/benchmark/run_all.sh                    # all frameworks
#   bash tests/benchmark/run_all.sh nozomi camino      # subset
#
# Run from the nozomi/ repo root with the venv activated.

set -euo pipefail

# --- Resolve repo root (parent of tests/) ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# --- Timestamped results dir (so previous sweeps are never overwritten) ---
# User can override via BENCH_RESULTS_DIR. Otherwise create a timestamped
# subfolder under tests/benchmark/results/ and point "latest" at it.
RESULTS_ROOT="$SCRIPT_DIR/results"
if [[ -z "${BENCH_RESULTS_DIR:-}" ]]; then
    TS="$(date +%Y-%m-%d_%H%M%S)"
    BENCH_RESULTS_DIR="$RESULTS_ROOT/$TS"
fi
mkdir -p "$BENCH_RESULTS_DIR"
# Refresh the "latest" symlink so aggregate.py defaults to this sweep.
ln -sfn "$BENCH_RESULTS_DIR" "$RESULTS_ROOT/latest"
export BENCH_RESULTS_DIR
echo "[bench] results dir: $BENCH_RESULTS_DIR"

# How many repeats — single source of truth is BenchConfig.n_repeats in
# tests/benchmark/config.py (overridable via BENCH_N_REPEATS env var, which
# config.py already honors).
N_REPEATS="$(python -c 'from tests.benchmark.config import CFG; print(CFG.n_repeats)')"
# Which compartment to seed walkers in: 
# Default is "intra" for the intra-axonal benchmark experiment (cylinder
# substrate, x-gradient perpendicular to fibers) where we compare against the
# van Gelderen analytic signal.
COMPARTMENT="${BENCH_COMPARTMENT:-intra}"
export BENCH_COMPARTMENT="$COMPARTMENT"
echo "[bench] compartment: $COMPARTMENT"
# Which GPU to use for the GPU frameworks (Nozomi, Disimpy). Precedence:
#   1. CUDA_VISIBLE_DEVICES (if already exported in the caller's shell)
#   2. $GPU env var
#   3. default "0"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU="$CUDA_VISIBLE_DEVICES"
else
    GPU="${GPU:-0}"
fi
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[bench] GPU (CUDA_VISIBLE_DEVICES): $GPU"

# Which frameworks to run
if [[ $# -eq 0 ]]; then
    FRAMEWORKS=(nozomi camino disimpy mcdc)
else
    FRAMEWORKS=("$@")
fi

echo "[bench] repo root: $REPO_ROOT"
echo "[bench] repeats:   $N_REPEATS"
echo "[bench] frameworks: ${FRAMEWORKS[*]}"

# Build substrate cache once.
python -m tests.benchmark.substrate

# Print substrate + analytic intra signal summary so every sweep's log is
# self-documenting.
python - <<PY
from tests.benchmark.config import CFG
from tests.benchmark.analytic import analytic_signals
print(f"[bench] substrate: R = {CFG.cylinder_radius_m*1e6:.2f} um (D = {CFG.cylinder_radius_m*2e6:.2f} um)"
      f", sep = {CFG.cylinder_sep_m*1e6:.3f} um, ICVF = {CFG.icvf:.4f}"
      f", box = {CFG.box_xy_m*1e6:.2f} x {CFG.box_xy_m*1e6:.2f} x {CFG.box_z_m*1e6:.2f} um")
print(f"[bench] sequence: delta = {CFG.delta_s*1e3} ms, Delta = {CFG.Delta_s*1e3} ms, "
      f"TE = {CFG.te_s*1e3} ms, bvals = {list(CFG.bvals_s_mm2)} s/mm^2")
print(f"[bench] MC: N = {CFG.n_walkers}, dt = {CFG.time_step_s*1e6:.2f} us, compartment = $COMPARTMENT")
S = analytic_signals(CFG)
print(f"[bench] van Gelderen analytic intra S = {[f'{s:.5f}' for s in S]}")
PY

for fw in "${FRAMEWORKS[@]}"; do
    echo "========== $fw =========="
    for r in $(seq 0 $((N_REPEATS - 1))); do
        echo "--- $fw repeat $r ---"
        case "$fw" in
            nozomi)
                python -m tests.benchmark.run_nozomi --repeat "$r" --compartment "$COMPARTMENT"
                ;;
            disimpy)
                python -m tests.benchmark.run_disimpy --repeat "$r" --compartment "$COMPARTMENT"
                ;;
            camino)
                python -m tests.benchmark.run_camino --repeat "$r" --compartment "$COMPARTMENT"
                ;;
            mcdc)
                python -m tests.benchmark.run_mcdc --repeat "$r" --compartment "$COMPARTMENT"
                ;;
            *)
                echo "unknown framework: $fw" >&2
                exit 2
                ;;
        esac
    done
done

echo "========== aggregate =========="
python -m tests.benchmark.aggregate --run-dir "$BENCH_RESULTS_DIR"
