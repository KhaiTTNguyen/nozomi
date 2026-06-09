"""
Configuration for the substrate-generation scalability experiment.

Goal: measure how substrate GENERATION (axon packing) scales in wall-clock time
and peak GPU/host memory as the substrate grows, by sweeping ONLY the number of
fibers (axons). No diffusion simulation is run here.

Fixed substrate = the Section 4.2 reference (d168-K200-beading0.3):
  experiment/setup/substrate/single_substrate/2026-03-22-bead0.3/
      d168-K200-beading0.3-substrate.json

Key choices:
  * Uses ``target_volume_fraction`` (NOT ``final_volume_fraction``) so the 2D
    init VF is set directly -> single feasible pass, no VF auto-tune, clean and
    reproducible per-size cost.
  * box_length_init = 0 -> box side L auto-sizes from N_fibers; L grows ~ sqrt(N).
  * No myelin (no g_ratio) -> outer fibers only.
"""

from __future__ import annotations
from pathlib import Path

# --- Paths --------------------------------------------------------------------
SCALABILITY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCALABILITY_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"               # per-size run.json checkpoints
SUBSTRATE_OUT_DIR = RESULTS_DIR / "substrates"  # generated substrate artifacts

# Repo root = .../nozomi  (tests/scalability/ -> parents[1])
REPO_ROOT = SCALABILITY_DIR.parents[1]

# --- Fixed substrate parameters (Section 4.2 reference) -----------------------
# Mirrors d168-K200-beading0.3-substrate.json, with target_volume_fraction kept
# (not final_volume_fraction) and num_fibers overridden per sweep point.
BASE_PARAMS: dict = {
    "orientation_shape_parameter": 200,
    "box_length_init": 0,                 # 0 -> auto from N_fibers
    "target_volume_fraction": 0.57,       # direct 2D init VF (no auto-tune)
    "num_fibers": 500,                    # OVERRIDDEN per sweep point
    "mean_diameter": 1.68,
    "sigma_diameter": 0.45,
    "dist_shape": 0.1,
    "space_buffer_starts_ends": 0.23,
    "spheres_spacing": 0.5,
    "space_buffer_repulse": 0.001,
    "w_overlap": 10,
    "w_curve": 3,
    "w_length": 3,
    "bead_spacing_mean": 5.70,
    "bead_spacing_stdv": 2.88,
    "bead_alpha_mean": 0.3,
    "bead_alpha_stdv": 0.18,
    "repeats": 1,                         # 1 generation per size
}

# --- Sweep --------------------------------------------------------------------
# Number of fibers per sweep point. Box side L scales ~ sqrt(N) at fixed ICVF and
# diameter, so N=500 reproduces the ~69 um reference cube. The driver ascends N
# and stops at the first size that exceeds the GPU-memory budget.
VERIFICATION_NUM_FIBERS = (100, 500)
FULL_NUM_FIBERS = (100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000)

# --- GPU memory safety (shared server) ----------------------------------------
# Keep TOTAL card usage <= this fraction so >= (1 - fraction) stays free for
# other users' processes on the shared GPU.
GPU_BUDGET_FRACTION = 0.80

# --- Plotting -----------------------------------------------------------------
# substrate_main() renders 3D/cross-section figures that are irrelevant to (and
# would dominate / destabilize) a timing+memory scaling measurement at large N.
# The runner disables them so only the generation cost is measured.
DISABLE_PLOTTING = True


def params_for(num_fibers: int) -> dict:
    """Return a copy of BASE_PARAMS with ``num_fibers`` overridden."""
    p = dict(BASE_PARAMS)
    p["num_fibers"] = int(num_fibers)
    return p


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, RUNS_DIR, SUBSTRATE_OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
