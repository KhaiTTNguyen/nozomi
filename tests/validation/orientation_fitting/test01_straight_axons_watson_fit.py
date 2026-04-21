"""Sanity check for the arc-length Watson-kappa fitting pipeline.

Constructs **straight** axons whose global orientations are drawn from a
Watson distribution with prescribed concentration ``K``. Because each axon is
a single straight line, the only source of local tangent dispersion is the
prescribed Watson distribution itself -> the arc-length fit *should* recover
``K_fit ~= K_designed``.

For each K in {200, 20, 10, 8}:
    1. Sample N direction vectors from Watson(mu=+z, K).
    2. For each direction, generate a chain of overlapping spheres spanning
       z in [-L/2, L/2] (so ``split_matrix_to_list`` can split on z==L/2).
    3. Plot the 3D substrate geometry.
    4. Run the production arc-length plotting (OD, FOD, achieved Watson
       samples) -> saves into figs/<datetime>/K_<K>/.

Run
---
    source sim_venv/bin/activate
    python tests/validation/orientation_fitting/test01_straight_axons_watson_fit.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Make the repository importable regardless of cwd.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]  # .../nozomi
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

import simulation_toolkit.toolkit_params as config_params
from simulation_toolkit.substrate_generator.helper.watson_distribution import (
    WatsonDistribution,
)
from simulation_toolkit.utils import orientation_plot


# -----------------------------------------------------------------------------
# Test configuration
# -----------------------------------------------------------------------------
K_VALUES = [200, 20, 10, 8]
NUM_AXONS = 2000
BOX_LENGTH = 100.0             # micrometres (arbitrary for this sanity check)
SPHERE_RADIUS = 0.5            # um
SPHERE_SPACING = 0.5           # um along the axon direction
RNG_SEED = 0


def _build_straight_axons(directions: np.ndarray,
                          L: float,
                          sphere_spacing: float,
                          sphere_radius: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Build a ``(total_spheres, 5)`` array of (x, y, z, r, fiber_id) spheres.

    Each axon is a straight chain of overlapping spheres spanning
    z in [-L/2, L/2]. Starting (x0, y0) are drawn uniformly in the box.

    The last sphere of every axon has z == L/2 exactly, so that
    ``common_utils.split_matrix_to_list`` (which splits on ``A[i, 2] == L/2``)
    correctly recovers one axon per direction.
    """
    blocks = []
    for fid, d in enumerate(directions):
        d = np.asarray(d, dtype=float)
        d = d / np.linalg.norm(d)
        # Enforce positive z direction so the axon traverses z=-L/2 -> +L/2.
        if d[2] <= 0:
            d = -d
        # Number of sphere steps along the axon so that z spans L.
        # Step along the direction is sphere_spacing; z increment per step is
        # sphere_spacing * dz. Choose n so that n * step_z ~ L.
        step_z = sphere_spacing * d[2]
        if step_z < 1e-6:
            # Near-equatorial direction; skip (would need too many spheres).
            continue
        n_steps = int(np.floor(L / step_z))
        # Parameterise along z so the last sphere lands exactly on z = +L/2.
        z = np.linspace(-L / 2, L / 2, n_steps + 1)
        # Random lateral offset so axons don't all pass through the origin.
        x0 = rng.uniform(-L / 2, L / 2)
        y0 = rng.uniform(-L / 2, L / 2)
        t = (z + L / 2) / d[2]  # arc-length along direction from start
        x = x0 + t * d[0]
        y = y0 + t * d[1]
        r = np.full_like(x, sphere_radius)
        fid_col = np.full_like(x, float(fid))
        blocks.append(np.stack([x, y, z, r, fid_col], axis=1))

    if not blocks:
        return np.empty((0, 5))
    return np.concatenate(blocks, axis=0)


def _plot_substrate_3d(xyz_r_fid: np.ndarray, save_path: str, title: str):
    """Quick 3D scatter of the straight-axon substrate."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    fiber_ids = np.unique(xyz_r_fid[:, 4])
    colors = cm.rainbow(np.linspace(0, 1, len(fiber_ids)))
    for c, fid in zip(colors, fiber_ids):
        mask = xyz_r_fid[:, 4] == fid
        pts = xyz_r_fid[mask]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=c, linewidth=0.7, alpha=0.8)
    L = BOX_LENGTH
    ax.set_xlim([-L / 2, L / 2])
    ax.set_ylim([-L / 2, L / 2])
    ax.set_zlim([-L / 2, L / 2])
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_zlabel('z (um)')
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def _run_one_K(K: float, out_root: Path, rng: np.random.Generator):
    print(f"\n=== K = {K} ===")
    K_folder = out_root / f"K_{int(K)}"
    K_folder.mkdir(parents=True, exist_ok=True)

    # 1. Sample directions from Watson(mu=+z, K).
    watson = WatsonDistribution(mu=np.array([0.0, 0.0, 1.0]), kappa=K)
    directions = watson.sample(NUM_AXONS)  # upper hemisphere by construction

    # 2. Build straight axons.
    xyz_r_fid = _build_straight_axons(
        directions=directions,
        L=BOX_LENGTH,
        sphere_spacing=SPHERE_SPACING,
        sphere_radius=SPHERE_RADIUS,
        rng=rng,
    )
    n_fibers = int(np.unique(xyz_r_fid[:, 4]).size)
    print(f"    built {n_fibers} straight axons, {xyz_r_fid.shape[0]} spheres total")

    # 3. Plot substrate geometry.
    _plot_substrate_3d(
        xyz_r_fid,
        save_path=str(K_folder / f"substrate_3d_K{int(K)}.png"),
        title=f"Straight axons from Watson(K={K}), n={n_fibers}",
    )

    # 4. Wire up config for the production plot function and run it.
    config_params.BOX_LENGTH = float(BOX_LENGTH)
    config_params.SUBSTRATE_OUTPUT_FOLDER_PATH = str(K_folder)
    config_params.ORIENTATION_SHAPE_PARAM = float(K)

    fit = orientation_plot.plot_along_axon_OD_arclength(
        xyz_r_fid, optimized=True,
    )
    print(f"    K_designed={K}  K_fit={fit['kappa']:.3f}  "
          f"ODI_fit={fit['ODI']:.5f}  n_tangents={fit['n_samples']}")
    return {
        'K_designed': float(K),
        'K_fit': float(fit['kappa']),
        'ODI_fit': float(fit['ODI']),
        'n_tangents': int(fit['n_samples']),
        'n_fibers': int(n_fibers),
    }


def main():
    rng = np.random.default_rng(RNG_SEED)
    # Also seed WatsonDistribution's numpy.random (it uses global RNG).
    np.random.seed(RNG_SEED)

    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_root = _HERE / 'figs' / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out_root}")

    rows = []
    for K in K_VALUES:
        rows.append(_run_one_K(K, out_root, rng))

    print("\n=== Summary ===")
    print(f"{'K_designed':<12} {'K_fit':<12} {'ODI_fit':<12} {'n_tangents':<10}")
    for r in rows:
        print(f"{r['K_designed']:<12g} {r['K_fit']:<12.3f} "
              f"{r['ODI_fit']:<12.5f} {r['n_tangents']:<10}")
    print(f"\nPlots saved under: {out_root}")


if __name__ == "__main__":
    main()
