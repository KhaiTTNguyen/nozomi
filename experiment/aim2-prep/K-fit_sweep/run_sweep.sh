#!/usr/bin/env bash
# lmax/smooth sweep for FOD glyph distinctness on the d=1.68 K=10/20/200 sets.
# Outputs are redirected under K-fit_sweep so the real data folders stay clean.
set -u

cd "$(dirname "$0")/../../.." || exit 1   # -> nozomi repo root
# shellcheck disable=SC1091
source sim_venv/bin/activate 2>/dev/null

OUT_ROOT="experiment/aim2-prep/K-fit_sweep"
DATA="experiment/aim2-prep/data/2026-03-22_bead_05"

declare -a ROOTS=(
  "$DATA/bead0.5_d1.68_OD10_initVF0.41_500axons/2026-03-22_23-44_d1.68_K10_ODI_0.0635_bead_0.5_500fibers"
  "$DATA/bead0.5_d1.68_OD20_initVF0.435_500axons/2026-03-22_23-44_d1.68_K20_ODI_0.0318_bead_0.5_500fibers"
  "$DATA/bead0.5_d1.68_OD200_initVF0.465_500axons/2026-03-22_23-44_d1.68_K200_ODI_0.0032_bead_0.5_500fibers"
)
declare -a LMAX=(10 15 20)
declare -a SMOOTH=(0.2 0.5 1.0)

for root in "${ROOTS[@]}"; do
  for lmax in "${LMAX[@]}"; do
    for smooth in "${SMOOTH[@]}"; do
      echo "##### $(date '+%H:%M:%S')  root=$(basename "$root")  lmax=$lmax  smooth=$smooth"
      python -m simulation_toolkit.cli.reprocess_substrate_orientation \
        --root "$root" --out-root "$OUT_ROOT" --lmax "$lmax" --smooth "$smooth"
    done
  done
done
echo "##### SWEEP DONE $(date '+%H:%M:%S')"
