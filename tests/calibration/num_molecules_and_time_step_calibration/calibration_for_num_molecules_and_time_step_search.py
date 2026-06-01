import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.special import jnp_zeros
from scipy.interpolate import interp1d
import os
import pycuda.gpuarray as gpuarray
import numpy as np
import logging
import time
import torch
import torch.optim as optim

# Suppress matplotlib font substitution messages
logging.getLogger('matplotlib.mathtext').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

import simulation_toolkit.simulation_engine.diffsim3d as ds3
import simulation_toolkit.simulation_engine.geometry as geom
import simulation_toolkit.toolkit_params as config_params

# ---------------------------------------------------------------------------
# Analytical D(t) functions
# ---------------------------------------------------------------------------

def calculateDanalytical_longtime(a, D0):
    """Analytical intra-axonal radial D(t) (Burcaw 2015 / Stepišnik 1993)."""
    t_values = np.logspace(-4, 2, 100)
    beta_1k = jnp_zeros(1, 100)

    def D_t(t):
        term1 = a**2 / (4 * t)
        sum_term = np.sum(
            np.exp(-beta_1k**2 * D0 * t / a**2) / (beta_1k**2 * (beta_1k**2 - 1))
        )
        return term1 - (2 * a**2 / t) * sum_term

    D_values = [D_t(t) for t in t_values]
    return t_values, D_values


def calculateDanalytical_multi_cylinder(primary_radii, D0, weights=None):
    """Volume-weighted analytical D(t) for a multi-cylinder substrate.

    By default each cylinder i contributes with weight proportional to its
    cross-sectional area r_i^2 (volume-weighted, assuming equal cylinder
    lengths):

        D(t) = sum_i w_i * D_i(t),   w_i = r_i^2 / sum_j r_j^2

    Pass ``weights`` to override with realised per-cylinder spin fractions
    N_i / N (Phase 3). Weights are normalised to sum to 1 internally.

    Uses the Burcaw/Stepišnik long-time solution per cylinder. Computation is
    fully vectorised over radii, time points, and Bessel roots.

    Parameters
    ----------
    primary_radii : ndarray  shape (N,)  cylinder radii in μm
    D0 : float               free diffusivity in μm²/ms
    weights : ndarray, optional, shape (N,)
        Per-cylinder weights. If None, defaults to r_i² / Σ r_j².

    Returns
    -------
    t_values : ndarray  shape (T,)   time points in ms
    D_weighted : ndarray shape (T,)  weighted D(t) in μm²/ms
    """
    t_values = np.logspace(-4, 2, 100)           # (T,)
    beta_1k = jnp_zeros(1, 100)                  # (K,)

    r = primary_radii[:, np.newaxis, np.newaxis]  # (N, 1, 1)
    t = t_values[np.newaxis, :, np.newaxis]       # (1, T, 1)
    b = beta_1k[np.newaxis, np.newaxis, :]        # (1, 1, K)

    term1 = r[:, :, 0] ** 2 / (4.0 * t[:, :, 0])              # (N, T)
    exp_term = np.exp(-b**2 * D0 * t / r**2)                   # (N, T, K)
    sum_term = np.sum(exp_term / (b**2 * (b**2 - 1)), axis=2)  # (N, T)
    D_all = term1 - (2.0 * r[:, :, 0] ** 2 / t[:, :, 0]) * sum_term  # (N, T)

    if weights is None:
        weights = primary_radii ** 2
    else:
        weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    D_weighted = np.sum(weights[:, np.newaxis] * D_all, axis=0)  # (T,)

    return t_values, D_weighted


# ---------------------------------------------------------------------------
# Substrate generation
# ---------------------------------------------------------------------------

def axon_gamma_dist_gen(n_axons, axon_area_fraction, kappa, theta_um,
                        rng_seed=0, space_buffer_fraction=0.25,
                        restart_at=10, max_restarts=20):
    """Pack n_axons parallel cylinders with Gamma(kappa, theta_um μm) diameters.

    Uses L-BFGS packing with periodic boundary conditions, mirroring
    Init2D.create_opt_2D_packing_with_auto_restart.  Only the (x, y) positions
    are optimised — radii and fiber-ids are kept as fixed non-gradient tensors so
    that L-BFGS only tracks N×2 active dimensions and its Hessian estimate stays
    well-conditioned.  Fresh random positions are tried after ``restart_at``
    L-BFGS steps without convergence.

    Parameters
    ----------
    n_axons           : number of cylinders to place
    axon_area_fraction: target 2D area fraction in (0, 1)
    kappa, theta_um   : Gamma shape / scale for diameters (μm)
    rng_seed              : reproducibility seed
    space_buffer_fraction : buffer = this × mean_radius, scales with the distribution
                            (default 0.25, matching Init2D convention)
    restart_at            : restart after this many L-BFGS steps (Init2D = 10)
    max_restarts          : maximum number of random restarts

    Returns
    -------
    x0, y0 : ndarray, centres in [-l_box/2, l_box/2] (μm)
    r0     : ndarray, radii (μm)
    l_box  : float, box side length (μm)
    """
    from simulation_toolkit.substrate_generator.helper import CollisionDetection2D as CD

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(rng_seed)
    axon_diameters = rng.gamma(shape=kappa, scale=theta_um, size=n_axons).astype(np.float32)

    # Dynamic space_buffer: scales with the mean radius of the actual sample,
    # matching Init2D's convention where the buffer is proportional to mean_diameter.
    mean_radius   = float(np.mean(axon_diameters / 2.0))
    space_buffer  = space_buffer_fraction * mean_radius
    print(f"  space_buffer = {space_buffer_fraction} × mean_radius({mean_radius:.4f} μm) "
          f"= {space_buffer:.4f} μm")

    area = float(np.sum(np.pi * (axon_diameters / 2.0) ** 2)) / axon_area_fraction
    l_box = float(np.sqrt(area))

    # Fixed (non-optimised) tensors on device — kept outside the gradient graph
    ra_dev  = torch.tensor(axon_diameters / 2.0, device=device)   # (N,) no grad
    fid_dev = torch.arange(n_axons, dtype=torch.float32, device=device)  # (N,) no grad

    def _init_pos(seed):
        """Random (N,2) starting positions on device."""
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        return (torch.rand(n_axons, 2, generator=gen) * l_box - l_box / 2.0).to(device)

    def _wrap_pbc(pos_xy):
        """Add periodic ghost copies for disks that cross the box boundary.

        pos_xy : (M, 2) tensor, may carry gradients for the primary N disks.
        ra_dev / fid_dev : fixed (N,) tensors — only primary disks, no gradients.

        Key rule: evaluate BOTH x-conditions on the original N-entry array before
        any extension, and BOTH y-conditions on the x-extended array before any
        y-extension.  This prevents a ghost from satisfying the opposite-side
        condition and being duplicated (dm=0 ⟹ NaN gradient).
        """
        xa  = pos_xy[:, 0]   # (N,) with grad
        ya  = pos_xy[:, 1]   # (N,) with grad
        ra  = ra_dev          # (N,) no grad
        fid = fid_dev         # (N,) no grad

        # x-axis — both masks on the ORIGINAL N entries
        ix  = torch.where(xa - ra < -l_box / 2)[0]   # left edge crosses left wall
        ixL = torch.where(xa + ra >  l_box / 2)[0]   # right edge crosses right wall

        xa  = torch.cat([xa,  xa[ix]  + l_box, xa[ixL]  - l_box])
        ya  = torch.cat([ya,  ya[ix],           ya[ixL]          ])
        ra  = torch.cat([ra,  ra[ix],           ra[ixL]          ])
        fid = torch.cat([fid, fid[ix],          fid[ixL]         ])

        # y-axis — both masks on the x-extended array (covers corner ghosts)
        iy  = torch.where(ya - ra < -l_box / 2)[0]   # bottom edge crosses bottom wall
        iyL = torch.where(ya + ra >  l_box / 2)[0]   # top edge crosses top wall

        xa  = torch.cat([xa,  xa[iy],           xa[iyL]          ])
        ya  = torch.cat([ya,  ya[iy]  + l_box, ya[iyL]  - l_box])
        ra  = torch.cat([ra,  ra[iy],           ra[iyL]          ])
        fid = torch.cat([fid, fid[iy],          fid[iyL]         ])

        return xa, ya, ra, fid

    def _count_overlaps(pos_xy_det):
        xa, ya, ra, fid = _wrap_pbc(pos_xy_det)
        cpos = torch.stack([xa, ya]).T
        cset = CD.detect_collision(cpos, ra, fid, l_box, l_box, 0.0, device)
        return int(cset.shape[0])

    # ── L-BFGS loop — matches Init2D.create_opt_2D_packing_with_auto_restart ─
    print('------ Start L-BFGS 2D packing ------')
    final_pos = None
    for attempt in range(max_restarts):
        print(f"  Attempt {attempt + 1}/{max_restarts} (new random initialisation)...")
        pos_xy = _init_pos(rng_seed + attempt).contiguous()
        pos_xy.requires_grad_(True)
        optimizer = optim.LBFGS([pos_xy], lr=0.1, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            c_xa, c_ya, c_ra, c_fid = _wrap_pbc(pos_xy)
            c_pos = torch.stack([c_xa, c_ya]).T.contiguous()
            # c_ra / c_fid carry no gradients (ra_dev / fid_dev are non-grad)
            cset = CD.detect_collision(c_pos, c_ra, c_fid, l_box, l_box, space_buffer, device)
            if cset.shape[0] == 0:
                # Zero loss connected to pos_xy so grad is correctly set to 0
                loss = (c_pos * 0.0).sum()
                loss.backward()
                return loss
            left, right = cset[:, 0], cset[:, 1]
            dp   = c_pos[left] - c_pos[right]
            dm   = torch.linalg.norm(dp, dim=1, keepdim=True).clamp(min=1e-8)
            rm   = (c_ra[left] + c_ra[right]).unsqueeze(1)
            loss = torch.sum(torch.clamp(rm + space_buffer - dm, min=0.0) ** 2)
            loss.backward()
            return loss

        num_iter = 0
        while True:
            with torch.no_grad():
                n_ov = _count_overlaps(pos_xy.detach())
            print(f"    Iteration {num_iter} . Overlaps {n_ov}")
            if n_ov == 0:
                print("    Final num_overlap: 0")
                final_pos = pos_xy.detach().cpu()
                break
            if num_iter == restart_at:
                print(f"    Overlaps at iteration {num_iter}. Restarting...")
                break
            optimizer.step(closure)
            num_iter += 1

        if final_pos is not None:
            break

    if final_pos is None:
        final_pos = pos_xy.detach().cpu()
        with torch.no_grad():
            remaining = _count_overlaps(final_pos.to(device))
        print(f"  [Warning] {remaining} overlaps remain after {max_restarts} restarts.")

    print('------ Done L-BFGS 2D packing ------')
    xy = final_pos.numpy()
    return xy[:, 0], xy[:, 1], axon_diameters / 2.0, l_box


def build_edge_ghost_cylinders(centers_xy, radii, lx, ly, edge_distance):
    """Return (centers, radii, n_primary) with primaries first, ghosts second.

    The first ``n_primary`` entries are the original cylinders (in input order);
    everything after that index is a periodic ghost copy of a primary that lies
    within ``edge_distance`` of a box boundary. Putting primaries first lets the
    caller seed spins only into primaries via contiguous indices [0, n_primary).
    """
    x_min, x_max = -lx / 2.0, lx / 2.0
    y_min, y_max = -ly / 2.0, ly / 2.0

    centers_arr = np.asarray(centers_xy, dtype=float)
    radii_arr = np.asarray(radii, dtype=float)
    n_primary = len(radii_arr)

    primary_centers = [tuple(c) for c in centers_arr]
    primary_radii_list = list(radii_arr)

    ghost_centers = []
    ghost_radii = []

    for (cx, cy), radius in zip(centers_arr, radii_arr):
        x_shifts = [0.0]
        y_shifts = [0.0]
        if (cx - x_min) <= edge_distance:
            x_shifts.append(lx)
        if (x_max - cx) <= edge_distance:
            x_shifts.append(-lx)
        if (cy - y_min) <= edge_distance:
            y_shifts.append(ly)
        if (y_max - cy) <= edge_distance:
            y_shifts.append(-ly)

        for dx in x_shifts:
            for dy in y_shifts:
                if dx == 0.0 and dy == 0.0:
                    continue  # primary already added
                ghost_centers.append((cx + dx, cy + dy))
                ghost_radii.append(radius)

    all_centers = np.asarray(primary_centers + ghost_centers, dtype=float)
    all_radii = np.asarray(primary_radii_list + ghost_radii, dtype=float)
    return all_centers, all_radii, n_primary


def build_multi_cylinder_geometry(
    centers_xy, radii, lx, ly, lz, D0, n_segments=None, extension_segments=5
):
    """Build a SimGeometry3D with z-aligned cylinders (no T2 weighting).

    Cylinders are added in input order, so if the caller passes primaries first
    then ghosts, structure indices [0, n_primary) correspond to primaries and
    [n_primary, nstructures) correspond to ghosts. Each cylinder is represented
    as a chain of overlapping spheres along z.

    n_segments : int or None
        Number of sphere centres along z per cylinder. If None (default), each
        cylinder's sphere spacing is set equal to its own radius (adjacent
        spheres just touch), guaranteeing no gaps. Large cylinders get fewer
        spheres; small cylinders get proportionally more. Pass an explicit int
        to override for all cylinders.
    """
    T2 = 1e10   # effectively infinite T2 — no T2 weighting
    rho = 1.0
    sg3 = geom.SimGeometry3D(lx, ly, lz, D0, T2, rho)

    total_spheres = 0
    for (cx, cy), radius in zip(centers_xy, radii):
        # Per-cylinder sphere count: spacing = radius guarantees the chain is
        # watertight (each sphere just reaches its neighbour). Uniform n_segments
        # wastes spheres on large cylinders and can leave gaps in small ones.
        if n_segments is not None:
            n_seg = n_segments
        else:
            n_seg = max(int(np.ceil(lz / (radius / 4.0))) + 1, 5)

        base_z = np.linspace(-lz / 2.0, lz / 2.0, num=n_seg)
        sx = np.full_like(base_z, cx, dtype=float)
        sy = np.full_like(base_z, cy, dtype=float)
        sz = base_z.copy()
        sr = np.full_like(base_z, radius, dtype=float)

        sz_ext = np.concatenate(
            (sz[-extension_segments:] - lz, sz, sz[:extension_segments] + lz)
        )
        sx_ext = np.concatenate(
            (sx[-extension_segments:], sx, sx[:extension_segments])
        )
        sy_ext = np.concatenate(
            (sy[-extension_segments:], sy, sy[:extension_segments])
        )
        sr_ext = np.concatenate(
            (sr[-extension_segments:], sr, sr[:extension_segments])
        )

        total_spheres += len(sz_ext)
        cylinder = geom.Structure3D(sx_ext, sy_ext, sz_ext, sr_ext, D0, T2, rho)
        sg3.add_structure(cylinder)

    print(f"  [geometry] {len(radii)} cylinders, {total_spheres} total spheres "
          f"(mean {total_spheres / len(radii):.1f} per cylinder)")
    return sg3


def _plot_substrate_geometry(centers_xy, radii, lx, ly, output_dir):
    """Plot axon cross-section geometry (primary box in blue, ghosts in gray)."""
    x = centers_xy[:, 0]
    y = centers_xy[:, 1]
    r = np.asarray(radii)

    in_box = (
        (x >= -lx / 2.0) & (x <= lx / 2.0)
        & (y >= -ly / 2.0) & (y <= ly / 2.0)
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    for xi, yi, ri in zip(x[~in_box], y[~in_box], r[~in_box]):
        ax.add_patch(
            Circle((xi, yi), ri, fill=False, linewidth=0.5,
                   edgecolor="tab:gray", alpha=0.35)
        )
    for xi, yi, ri in zip(x[in_box], y[in_box], r[in_box]):
        ax.add_patch(
            Circle((xi, yi), ri, fill=False, linewidth=0.8,
                   edgecolor="tab:blue", alpha=0.9)
        )

    ax.plot(
        [-lx / 2, lx / 2, lx / 2, -lx / 2, -lx / 2],
        [-ly / 2, -ly / 2, ly / 2, ly / 2, -ly / 2],
        "k-", linewidth=1.5,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.6 * lx, 0.6 * lx)
    ax.set_ylim(-0.6 * ly, 0.6 * ly)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_title(
        "Calibration substrate – Gamma(κ, θ) cylinder radii\n"
        "(blue: primary box, gray: periodic ghosts)"
    )
    ax.text(
        0.02, 0.02,
        f"Primary cylinders: {np.count_nonzero(in_box)}\n"
        f"Total incl. ghosts: {len(x)}",
        transform=ax.transAxes, fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "calibration_substrate_geometry.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase-1 diagnostic: realised seeding distribution
# ---------------------------------------------------------------------------

def _bin_spins_per_cylinder(spins_xy, centers_xy, radii):
    """Return (counts_per_cyl, in_any_mask) using a point-in-disc test.

    Cylinders are z-aligned, so (x, y) point-in-disc is sufficient.
    Assumes no overlap between primaries (true for the calibration substrate).
    """
    n = spins_xy.shape[0]
    counts = np.zeros(len(radii), dtype=np.int64)
    in_any = np.zeros(n, dtype=bool)
    for i, ((cx, cy), r) in enumerate(zip(centers_xy, radii)):
        d2 = (spins_xy[:, 0] - cx) ** 2 + (spins_xy[:, 1] - cy) ** 2
        mask = d2 < r * r
        counts[i] = int(mask.sum())
        in_any |= mask
    return counts, in_any


def diagnose_seeding_distribution(
    sg3, primary_centers_xy, primary_radii,
    ghost_centers_xy, ghost_radii,
    n_primary, n_probe_spins, output_dir,
):
    """Seed a small batch of spins into primaries only and report the realised
    per-cylinder population vs theoretical volume weights r_i² / Σ r_j².

    Writes a CSV (realised_seed_weights.csv) for downstream use in Phase 3.
    """
    print("\n[Diagnostic] Probing realised seeding distribution "
          f"({n_probe_spins} spins, primaries only)...")
    probe = ds3.DiffSim3d(sg3, int(n_probe_spins))
    probe.set_segments(nsegx=20, nsegy=20, nsegz=20)
    probe.setup(structures=list(range(n_primary)))

    # sim.spins shape: (3, nspin) — rows are x, y, z
    spins_arr = np.asarray(probe.spins, dtype=np.float64)
    n = int(n_probe_spins)
    spins_xy = np.column_stack((spins_arr[0, :n], spins_arr[1, :n]))

    counts_prim, in_prim = _bin_spins_per_cylinder(
        spins_xy, primary_centers_xy, primary_radii,
    )
    if len(ghost_radii) > 0:
        counts_ghost, in_ghost = _bin_spins_per_cylinder(
            spins_xy, ghost_centers_xy, ghost_radii,
        )
    else:
        counts_ghost = np.zeros(0, dtype=np.int64)
        in_ghost = np.zeros(n, dtype=bool)

    n_in_prim = int(in_prim.sum())
    n_in_ghost = int(in_ghost.sum())
    n_outside = int((~(in_prim | in_ghost)).sum())

    theoretical = primary_radii ** 2 / np.sum(primary_radii ** 2)
    realised = counts_prim / max(counts_prim.sum(), 1)

    diff = realised - theoretical
    rel_err = np.where(theoretical > 0, diff / theoretical, 0.0)

    print(f"  Total probe spins:           {n}")
    print(f"  Spins inside any primary:    {n_in_prim} ({n_in_prim / n:.4%})")
    print(f"  Spins inside any ghost:      {n_in_ghost} ({n_in_ghost / n:.4%})")
    print(f"  Spins outside any cylinder:  {n_outside} ({n_outside / n:.4%})")
    print(f"  Per-primary weight statistics (realised vs theoretical r²/Σr²):")
    print(f"    mean |Δw|             = {np.mean(np.abs(diff)):.3e}")
    print(f"    max  |Δw|             = {np.max(np.abs(diff)):.3e}")
    print(f"    mean |Δw|/w_theory    = {np.mean(np.abs(rel_err)):.3%}")
    print(f"    median|Δw|/w_theory   = {np.median(np.abs(rel_err)):.3%}")

    csv_path = os.path.join(output_dir, "realised_seed_weights.csv")
    pd.DataFrame({
        "primary_idx": np.arange(n_primary),
        "radius_um": primary_radii,
        "count": counts_prim,
        "w_realised": realised,
        "w_theoretical_r2": theoretical,
        "abs_diff": np.abs(diff),
        "rel_err": rel_err,
    }).to_csv(csv_path, index=False)
    print(f"  [Diagnostic] Saved per-cylinder seeding stats to {csv_path}")

    # release probe resources
    del probe


def compute_realised_weights(sim, primary_centers_xy, primary_radii):
    """Return realised per-primary spin fractions w_i = N_i / N.

    Reads the initial spin positions from ``sim.spins`` (shape (3, nspins))
    and bins by (x, y) point-in-disc test against each primary cylinder.
    Returned weights are normalised so they sum to the fraction of spins
    that landed inside any primary (= 1.0 when primaries-only seeding is
    used, as in Phase 2).
    """
    spins_arr = np.asarray(sim.spins, dtype=np.float64)
    n = spins_arr.shape[1]
    spins_xy = np.column_stack((spins_arr[0], spins_arr[1]))
    counts, _ = _bin_spins_per_cylinder(
        spins_xy, primary_centers_xy, primary_radii,
    )
    return counts.astype(float) / max(n, 1)


# ---------------------------------------------------------------------------
# Numerical simulation
# ---------------------------------------------------------------------------

def calculateDnumerical(
    sg3, D0, molecules, time_step, total_sim_time=100, run_id=0,
    seed_structures=None,
    primary_centers_xy=None, primary_radii=None,
):
    """Run DiffSim3d on a pre-built geometry; return radial D(t) and weights.

    Seeds spins inside the structures listed in ``seed_structures`` (defaults
    to all structures). For multi-cylinder substrates with periodic ghost
    cylinders, pass ``seed_structures=list(range(n_primary))`` so spins are
    placed only in primary cylinders — ghosts remain in geometry as walls so
    PBC-wrapped spins still see the correct intra-axonal boundary.

    If ``primary_centers_xy`` and ``primary_radii`` are provided, the
    realised per-primary spin fractions are computed from the initial seed
    positions and returned for Phase-3 per-run analytical referencing.

    Returns
    -------
    difftime : ndarray
    D_numerical : ndarray            radial D(t) = (Dx + Dy) / 2
    realised_weights : ndarray or None  per-primary N_i / N (or None)
    """
    nt = int(total_sim_time / time_step)
    numsteps = nt + 1

    if seed_structures is None:
        seed_structures = list(np.arange(0, sg3.nstructures))

    sim = ds3.DiffSim3d(sg3, int(molecules))
    sim.set_segments(nsegx=20, nsegy=20, nsegz=20)
    sim.setup(structures=list(seed_structures))

    realised_weights = None
    if primary_centers_xy is not None and primary_radii is not None:
        realised_weights = compute_realised_weights(
            sim, primary_centers_xy, primary_radii,
        )

    Dxarray = gpuarray.zeros(numsteps, dtype=np.float32)
    Dyarray = gpuarray.zeros(numsteps, dtype=np.float32)
    Dzarray = gpuarray.zeros(numsteps, dtype=np.float32)
    Kx2array = gpuarray.zeros(numsteps, dtype=np.float32)
    Ky2array = gpuarray.zeros(numsteps, dtype=np.float32)
    Kz2array = gpuarray.zeros(numsteps, dtype=np.float32)
    Kx4array = gpuarray.zeros(numsteps, dtype=np.float32)
    Ky4array = gpuarray.zeros(numsteps, dtype=np.float32)
    Kz4array = gpuarray.zeros(numsteps, dtype=np.float32)

    difftime = np.zeros(numsteps)
    current_time = 0.0
    step_idx = 1

    while current_time < nt * time_step:
        sim.step(time_step)
        current_time += time_step
        sim.calculate_diffusion_coefficients_and_kurtoses(
            step_idx, current_time,
            Dxarray, Dyarray, Dzarray,
            Kx2array, Ky2array, Kz2array,
            Kx4array, Ky4array, Kz4array,
        )
        difftime[step_idx - 1] = current_time
        step_idx += 1

    Dxstep = np.array(Dxarray.get())[1:]
    Dystep = np.array(Dyarray.get())[1:]
    difftime = difftime[1:]

    D_numerical = (Dxstep + Dystep) / 2.0
    return difftime, D_numerical, realised_weights


# ---------------------------------------------------------------------------
# Error metric
# ---------------------------------------------------------------------------

def calculate_mae(D_numerical, D_long_interp, difftime, time_threshold=0.05):
    """MAE between numerical and interpolated analytical D(t) for t >= time_threshold."""
    valid_mask = difftime >= time_threshold
    if not np.any(valid_mask):
        print("No valid time points found for MAE calculation.")
        return np.nan

    difftime_filt = difftime[valid_mask]
    D_num_filt = D_numerical[valid_mask]
    D_ana_filt = D_long_interp(np.log10(difftime_filt))
    return np.mean(np.abs(D_num_filt - D_ana_filt))


# ---------------------------------------------------------------------------
# Output / plotting
# ---------------------------------------------------------------------------

def plot_numerical_vs_analytical_comparison(
    difftime, D_numerical, t_analytical, D_analytical,
    molecules, time_step, run_id, experiment_dir, time_threshold=0.05,
    primary_radii=None, D0=2.0, weights=None,
):
    """Save per-run D(t) comparison plot (simulated vs weighted analytical).

    If ``weights`` is given, the green short-time and orange long-time
    asymptotes use those weights too (consistent with the blue GPA curve).
    Otherwise they fall back to theoretical r_i² / Σ r_j².
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    mask = difftime >= time_threshold
    ax.semilogx(
        difftime[mask], D_numerical[mask], "o",
        markerfacecolor="none", markeredgecolor="red", markersize=8,
        alpha=0.6, label=r"$D_{\perp, \mathrm{MC}}(t)$",
    )

    # Volume-weighted short-time limit (green dashed)
    if primary_radii is not None:
        if weights is None:
            w = primary_radii ** 2
        else:
            w = np.asarray(weights, dtype=float)
        w = w / np.sum(w)
        # Volume-weighted S/V coefficient: C = (4/(3*d*sqrt(pi))) * 2 * sum(w_i / r_i), d=2
        C_vw = (4.0 / (3.0 * 2.0 * np.sqrt(np.pi))) * 2.0 * np.sum(w / primary_radii)
        t_cutoff = (1.0 / C_vw) ** 2 / D0
        t_short = np.logspace(-4, np.log10(0.95 * t_cutoff), 100)
        D_short = D0 * (1.0 - C_vw * np.sqrt(D0 * t_short))
        D_short = np.clip(D_short, 0, None)
        ax.semilogx(
            t_short, D_short, color="green", linestyle="dashed", linewidth=5,
            alpha=0.8, label=r"$D_{\perp, t \rightarrow 0}(t)$",
        )

    # GPA (volume-weighted long-time Bessel series) — blue solid
    ax.semilogx(
        t_analytical, D_analytical, "b-", linewidth=5,
        alpha=0.8, label=r"$D_{\perp, \mathrm{GPA}}(t)$",
    )

    # Volume-weighted long-time asymptote D ~ <r²>_w / (4t) (orange dashed)
    if primary_radii is not None:
        r2_vw = np.sum(w * primary_radii**2)
        t_asymptotic = np.logspace(np.log10(0.3), 2, 100)
        D_asymptotic = r2_vw / (4.0 * t_asymptotic)
        ax.semilogx(
            t_asymptotic, D_asymptotic, color="orange", linestyle="dashed", linewidth=5,
            alpha=0.8, label=r"$D_{\perp, t \rightarrow \infty}(t)$",
        )

    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_xlabel("Time (ms)", fontsize=25)
    ax.set_ylabel(r"$D_{\perp}(t)$ (μm²/ms)", fontsize=25)
    ax.tick_params(axis="both", which="major", labelsize=17)
    ax.set_ylim(0, D0 * 1.1)
    ax.legend(fontsize=25, framealpha=0.9)
    plt.tight_layout()

    plot_path = os.path.join(experiment_dir, f"cylinder_analysis_run_{run_id}.png")
    plt.savefig(plot_path, dpi=500, bbox_inches="tight")
    plt.close()


def save_experiment_data(
    difftime, D_numerical, t_analytical, D_analytical,
    molecules, time_step, run_id, base_dir, mae, time_threshold=0.05,
    primary_radii=None, D0=2.0, weights=None,
):
    """Save per-run comparison plot."""
    experiment_dir = os.path.join(
        base_dir, f"molecules_{int(molecules)}_timestep_{time_step}"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    plot_numerical_vs_analytical_comparison(
        difftime, D_numerical, t_analytical, D_analytical,
        molecules, time_step, run_id, experiment_dir, time_threshold,
        primary_radii=primary_radii, D0=D0, weights=weights,
    )


def save_summary_results(summary_results, base_dir, kappa, theta_um, time_threshold):
    """Save summary results to CSV."""
    df = pd.DataFrame(summary_results)
    csv_path = os.path.join(
        base_dir,
        f"cylinder_validation_summary_kappa{kappa}_theta{theta_um}"
        f"_threshold{time_threshold}.csv",
    )
    df.to_csv(csv_path, index=False)
    print(f"\nSummary results saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------

RUNS_CSV_NAME = "runs.csv"
RUNS_CSV_COLUMNS = [
    "molecules", "time_step", "run_id",
    "mae", "computation_time", "timestamp",
]


def make_or_resume_output_dir(resume_path=None):
    """Return an output directory.

    If ``resume_path`` is None, a fresh timestamped directory is created.
    Otherwise the given directory is reused (relative paths are resolved
    inside the standard figs/ folder).
    """
    base = "./tests/calibration/num_molecules_and_time_step_calibration/figs"
    if resume_path is None:
        path = os.path.join(
            base,
            "cylinder_validation_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        os.makedirs(path, exist_ok=True)
        print(f"[checkpoint] Fresh output dir: {path}")
        return path

    path = resume_path if (os.path.isabs(resume_path) or os.path.isdir(resume_path)) \
        else os.path.join(base, resume_path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"--resume directory not found: {path}")
    print(f"[checkpoint] Resuming into: {path}")
    return path


def load_completed_runs(runs_csv_path):
    """Return a set of (molecules, time_step, run_id) triples already recorded.

    NaN-MAE rows are still considered completed (no auto-retry); delete those
    rows from runs.csv to force a re-run.
    """
    if not os.path.isfile(runs_csv_path):
        return set()
    try:
        df = pd.read_csv(runs_csv_path)
    except Exception as e:
        print(f"[checkpoint] Could not read existing {runs_csv_path}: {e}")
        return set()
    completed = set(
        zip(
            df["molecules"].astype(int),
            df["time_step"].astype(float),
            df["run_id"].astype(int),
        )
    )
    print(f"[checkpoint] Loaded {len(completed)} completed runs from {runs_csv_path}")
    return completed


def append_run_record(runs_csv_path, record):
    """Append one row to the per-run checkpoint CSV (writes header if missing)."""
    write_header = not os.path.isfile(runs_csv_path)
    pd.DataFrame([record], columns=RUNS_CSV_COLUMNS).to_csv(
        runs_csv_path, mode="a", header=write_header, index=False,
    )


def aggregate_summary_from_runs(runs_csv_path, time_threshold):
    """Aggregate per-run records into the summary-row format used by
    ``create_analysis_plots`` / ``save_summary_results``.
    """
    if not os.path.isfile(runs_csv_path):
        return []
    df = pd.read_csv(runs_csv_path)
    summary = []
    for (mol, ts), grp in df.groupby(["molecules", "time_step"]):
        valid = grp["mae"].dropna()
        if valid.empty:
            continue
        ct = grp["computation_time"].dropna()
        summary.append({
            "molecules": int(mol),
            "time_step": float(ts),
            "time_threshold": time_threshold,
            "mae": float(valid.mean()),
            "std_mae": float(valid.std(ddof=0)),
            "min_mae": float(valid.min()),
            "max_mae": float(valid.max()),
            "mean_computation_time": float(ct.mean()) if not ct.empty else float("nan"),
            "std_computation_time": float(ct.std(ddof=0)) if not ct.empty else float("nan"),
            "n_successful_runs": int(valid.shape[0]),
        })
    return summary


def create_analysis_plots(
    summary_results, base_dir, kappa, theta_um, time_threshold, total_time_hours
):
    """Three-panel figure: MAE vs time step | computation time log | computation time linear."""
    df = pd.DataFrame(summary_results)

    molecules_list = sorted(df["molecules"].unique())
    time_steps_list = sorted(df["time_step"].unique())
    colors_mol = plt.cm.plasma(np.linspace(0, 1, len(molecules_list)))
    colors_ts = plt.cm.viridis(np.linspace(0, 1, len(time_steps_list)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    # ── Plot 1: MAE vs Time Step (log-log) ──────────────────────────────────
    for i, mol in enumerate(molecules_list):
        data = df[df["molecules"] == mol].sort_values("time_step")
        ax1.loglog(
            data["time_step"], data["mae"], "s-",
            color=colors_mol[i], label=f"{mol:.0e}", linewidth=2, markersize=6,
        )
        ax1.fill_between(
            data["time_step"],
            np.maximum(data["mae"] - data["std_mae"], 1e-12),
            data["mae"] + data["std_mae"],
            color=colors_mol[i], alpha=0.15,
        )
    ax1.set_xlabel("Time Step (ms)", fontsize=15)
    ax1.set_ylabel("MAE (μm²/ms)", fontsize=15)
    ax1.set_title("MAE vs Time Step", fontsize=15)
    ax1.legend(title="Molecules", ncol=2, fontsize=11)
    ax1.grid(True, alpha=0.4)
    ax1.tick_params(axis="both", which="major", labelsize=13)
    ax1.tick_params(axis="both", which="minor", labelsize=11)

    # ── Plot 2: Computation Time vs Molecules (log-log) ──────────────────────
    for i, ts in enumerate(time_steps_list):
        data = df[df["time_step"] == ts].sort_values("molecules")
        ax2.loglog(
            data["molecules"], data["mean_computation_time"], "o-",
            color=colors_ts[i], label=f"dt={ts}", linewidth=2, markersize=6,
        )
    ax2.set_xlabel("Number of Molecules", fontsize=15)
    ax2.set_ylabel("Computation Time (s)", fontsize=15)
    ax2.set_title("Computation Time vs Molecules (log scale)", fontsize=15)
    ax2.legend(title="Time Step (ms)", ncol=2, fontsize=11)
    ax2.grid(True, alpha=0.4)
    ax2.tick_params(axis="both", which="major", labelsize=13)
    ax2.tick_params(axis="both", which="minor", labelsize=11)

    # ── Plot 3: Computation Time vs Molecules (linear) ────────────────────────
    for i, ts in enumerate(time_steps_list):
        data = df[df["time_step"] == ts].sort_values("molecules")
        ax3.plot(
            data["molecules"], data["mean_computation_time"], "o-",
            color=colors_ts[i], label=f"dt={ts}", linewidth=2, markersize=6,
        )
    ax3.set_xlabel("Number of Molecules", fontsize=15)
    ax3.set_ylabel("Computation Time (s)", fontsize=15)
    ax3.set_title("Computation Time vs Molecules (linear scale)", fontsize=15)
    ax3.legend(title="Time Step (ms)", ncol=2, fontsize=11)
    ax3.grid(True, alpha=0.4)
    ax3.tick_params(axis="both", which="major", labelsize=13)

    plt.suptitle(
        f"Calibration substrate: Gamma(κ={kappa}, θ={theta_um} μm)  |  "
        f"Total runtime: {total_time_hours:.2f} hrs",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    plot_path = os.path.join(
        base_dir,
        f"analytical_validation_analysis_kappa{kappa}_theta{theta_um}"
        f"_threshold{time_threshold}_comptime{total_time_hours:.2f}hrs.png",
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nAnalysis plots saved to: {plot_path}")


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------

def run_validation_study(resume_path=None):
    """Run num_molecules × time_step calibration on a Gamma-distributed cylinder substrate.

    If ``resume_path`` is given (path relative to figs/ or absolute), reuse
    that output directory and skip any (molecules, time_step, run_id) triples
    already recorded in ``runs.csv``.
    """
    st = time.time()

    # Physical parameters
    D0 = 2.0              # μm²/ms
    total_sim_time = 100  # ms
    time_threshold = 0.35 # ms (lower bound for MAE window)

    # Gamma distribution for axon diameters: Ŵ(κ, θ), θ in μm
    # θ = 4.5×10⁻⁷ m = 0.45 μm  →  mean diameter = κ × θ = 1.8 μm
    kappa = 4.0
    theta_um = 0.45
    n_axons = 500
    axon_area_fraction = 0.65

    # Calibration grid
    molecules_values = [int(1e6), int(5e5), int(1e5), int(5e4), int(1e4)]
    time_step_values = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    n_repeats = 5

    # molecules_values = [int(5e5), int(1e4)]
    # time_step_values = [0.0005, 0.005, 0.01]
    # n_repeats = 5

    # Create / resume output directory
    config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH = make_or_resume_output_dir(
        resume_path,
    )
    runs_csv_path = os.path.join(
        config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH, RUNS_CSV_NAME,
    )
    completed_runs = load_completed_runs(runs_csv_path)

    # ── 1. Generate substrate (cached) ──────────────────────────────────────
    substrate_path = os.path.join(
        config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH, "substrate.npz",
    )
    if os.path.isfile(substrate_path):
        print(f"\n[checkpoint] Loading cached substrate from {substrate_path}")
        d = np.load(substrate_path)
        x_centers = d["x_centers"]
        y_centers = d["y_centers"]
        primary_radii = d["primary_radii"]
        lxy = float(d["lxy"])
    else:
        print("Generating Gamma-distributed cylinder substrate (L-BFGS packing)...")
        x_centers, y_centers, primary_radii, lxy = axon_gamma_dist_gen(
            n_axons=n_axons,
            axon_area_fraction=axon_area_fraction,
            kappa=kappa,
            theta_um=theta_um,
            rng_seed=42,
        )
        np.savez(
            substrate_path,
            x_centers=x_centers, y_centers=y_centers,
            primary_radii=primary_radii, lxy=np.float64(lxy),
        )
        print(f"[checkpoint] Saved substrate to {substrate_path}")
    lx, ly, lz = lxy, lxy, 20.0
    primary_centers_xy = np.column_stack((x_centers, y_centers))

    centers_xy, radii, n_primary = build_edge_ghost_cylinders(
        centers_xy=primary_centers_xy,
        radii=primary_radii,
        lx=lx,
        ly=ly,
        edge_distance=lxy / 5.0,
    )

    print(f"  Primary cylinders placed: {len(primary_radii)}")
    print(f"  Total incl. edge ghosts:  {len(radii)}")
    print(f"  n_primary index range:    [0, {n_primary})")
    print(f"  Box side:                 {lxy:.2f} μm")
    print(f"  Mean radius (primary):    {primary_radii.mean():.3f} μm "
          f"± {primary_radii.std():.3f} μm")

    _plot_substrate_geometry(
        centers_xy=centers_xy,
        radii=radii,
        lx=lx, ly=ly,
        output_dir=config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH,
    )

    # ── 2. Build SimGeometry3D (once) ───────────────────────────────────────
    print("\nBuilding SimGeometry3D (this may take a moment)...")
    sg3 = build_multi_cylinder_geometry(centers_xy, radii, lx, ly, lz, D0)
    print(f"  nstructures: {sg3.nstructures}  (primaries=[0,{n_primary}), "
          f"ghosts=[{n_primary},{sg3.nstructures}))")

    # (Phase-1 200k-spin seeding-distribution probe disabled — realised weights
    # are now read per-run inside calculateDnumerical via compute_realised_weights.)

    # ── 3. Theoretical volume-weighted analytical D(t) (once, for reference) ──
    print("\nComputing volume-weighted analytical D(t) (theoretical w_i=r_i²/Σr_j²)...")
    t_analytical, D_analytical = calculateDanalytical_multi_cylinder(primary_radii, D0)
    print("  Done. (Per-run analytical with REALISED weights is computed below.)")

    # ── 4. Parameter sweep ──────────────────────────────────────────────────
    total_combinations = len(molecules_values) * len(time_step_values)
    current_combination = 0

    for molecules in molecules_values:
        for time_step in time_step_values:
            current_combination += 1
            print(f"\n{'='*40}")
            print(f"Combination {current_combination}/{total_combinations}  "
                  f"| molecules={molecules}  time_step={time_step} ms")

            for run_id in range(n_repeats):
                key = (int(molecules), float(time_step), int(run_id))
                if key in completed_runs:
                    print(f"  Run {run_id + 1}/{n_repeats} — already recorded, skipping.")
                    continue

                print(f"  Run {run_id + 1}/{n_repeats}", end=" ... ", flush=True)
                t0 = time.time()
                mae = np.nan
                elapsed = np.nan
                try:
                    difftime, D_numerical, realised_weights = calculateDnumerical(
                        sg3, D0, molecules, time_step, total_sim_time, run_id,
                        seed_structures=list(range(n_primary)),
                        primary_centers_xy=primary_centers_xy,
                        primary_radii=primary_radii,
                    )
                    elapsed = time.time() - t0
                    print(f"{elapsed:.1f} s")

                    # Phase 3: per-run analytical using REALISED weights N_i/N
                    t_ana_run, D_ana_run = calculateDanalytical_multi_cylinder(
                        primary_radii, D0, weights=realised_weights,
                    )
                    D_interp_run = interp1d(
                        np.log10(t_ana_run), D_ana_run,
                        bounds_error=False,
                        fill_value=(D_ana_run[0], D_ana_run[-1]),
                    )

                    mae = calculate_mae(D_numerical, D_interp_run, difftime, time_threshold)

                    save_experiment_data(
                        difftime, D_numerical, t_ana_run, D_ana_run,
                        molecules, time_step, run_id,
                        config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH,
                        mae, time_threshold,
                        primary_radii=primary_radii, D0=D0,
                        weights=realised_weights,
                    )
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"ERROR: {e}")

                # Persist this run immediately so a later crash does not lose it.
                append_run_record(runs_csv_path, {
                    "molecules": int(molecules),
                    "time_step": float(time_step),
                    "run_id": int(run_id),
                    "mae": mae,
                    "computation_time": elapsed,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                })
                completed_runs.add(key)

    # Aggregate from the persistent per-run CSV so resumed runs are included.
    summary_results = aggregate_summary_from_runs(runs_csv_path, time_threshold)
    total_hours = (time.time() - st) / 3600.0
    save_summary_results(
        summary_results,
        config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH,
        kappa, theta_um, time_threshold,
    )
    create_analysis_plots(
        summary_results,
        config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH,
        kappa, theta_um, time_threshold, total_hours,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-cylinder MC-vs-analytical calibration with resume support.",
    )
    parser.add_argument(
        "--resume", "-r", type=str, default=None,
        help=("Reuse an existing output directory (name relative to figs/ or "
              "absolute path). Already-recorded (molecules, time_step, run_id) "
              "triples in runs.csv are skipped."),
    )
    args = parser.parse_args()

    print("=======Starting Analytical Validation Study=======")
    run_validation_study(resume_path=args.resume)
    print("\nAnalytical validation study completed!")