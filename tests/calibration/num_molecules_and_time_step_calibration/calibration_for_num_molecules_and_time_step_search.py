import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

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


def calculateDanalytical_multi_cylinder(primary_radii, D0):
    """Volume-weighted analytical D(t) for a multi-cylinder substrate.

    Each cylinder i contributes with weight proportional to its cross-sectional
    area r_i^2 (volume-weighted, assuming equal cylinder lengths):

        D(t) = sum_i w_i * D_i(t),   w_i = r_i^2 / sum_j r_j^2

    Uses the Burcaw/Stepišnik long-time solution per cylinder. Computation is
    fully vectorised over radii, time points, and Bessel roots.

    Parameters
    ----------
    primary_radii : ndarray  shape (N,)  cylinder radii in μm
    D0 : float               free diffusivity in μm²/ms

    Returns
    -------
    t_values : ndarray  shape (T,)   time points in ms
    D_weighted : ndarray shape (T,)  volume-weighted D(t) in μm²/ms
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

    weights = primary_radii**2
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
    """Return primary + periodic ghost cylinders within edge_distance of each boundary."""
    x_min, x_max = -lx / 2.0, lx / 2.0
    y_min, y_max = -ly / 2.0, ly / 2.0

    expanded_centers = []
    expanded_radii = []

    for (cx, cy), radius in zip(
        np.asarray(centers_xy, dtype=float), np.asarray(radii, dtype=float)
    ):
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
                expanded_centers.append((cx + dx, cy + dy))
                expanded_radii.append(radius)

    return (
        np.asarray(expanded_centers, dtype=float),
        np.asarray(expanded_radii, dtype=float),
    )


def build_multi_cylinder_geometry(
    centers_xy, radii, lx, ly, lz, D0, n_segments=100, extension_segments=20
):
    """Build a SimGeometry3D with z-aligned cylinders (no T2 weighting).

    Each cylinder is represented as a chain of overlapping spheres along z.
    """
    T2 = 1e10   # effectively infinite T2 — no T2 weighting
    rho = 1.0
    sg3 = geom.SimGeometry3D(lx, ly, lz, D0, T2, rho)

    base_z = np.linspace(-lz / 2.0, lz / 2.0, num=n_segments)

    for (cx, cy), radius in zip(centers_xy, radii):
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

        cylinder = geom.Structure3D(sx_ext, sy_ext, sz_ext, sr_ext, D0, T2, rho)
        sg3.add_structure(cylinder)

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
# Numerical simulation
# ---------------------------------------------------------------------------

def calculateDnumerical(sg3, D0, molecules, time_step, total_sim_time=100, run_id=0):
    """Run DiffSim3d on a pre-built geometry; return radial D(t).

    Seeds all spins inside all cylinder structures (intra-axonal only).
    Returns radial (transverse) D(t) = (Dx + Dy) / 2.
    """
    nt = int(total_sim_time / time_step)
    numsteps = nt + 1

    sim = ds3.DiffSim3d(sg3, int(molecules))
    sim.set_segments(nsegx=20, nsegy=20, nsegz=20)
    sim.setup(structures=list(np.arange(0, sg3.nstructures)))

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
    return difftime, D_numerical


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
    primary_radii=None, D0=2.0,
):
    """Save per-run D(t) comparison plot (simulated vs volume-weighted analytical)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    mask = difftime >= time_threshold
    ax.semilogx(
        difftime[mask], D_numerical[mask], "o",
        markerfacecolor="none", markeredgecolor="red", markersize=8,
        alpha=0.6, label=r"$D_{\perp, \mathrm{MC}}(t)$",
    )

    # Volume-weighted short-time limit (green dashed)
    if primary_radii is not None:
        weights = primary_radii**2 / np.sum(primary_radii**2)
        # Volume-weighted S/V coefficient: C = (4/(3*d*sqrt(pi))) * 2 * sum(w_i / r_i), d=2
        C_vw = (4.0 / (3.0 * 2.0 * np.sqrt(np.pi))) * 2.0 * np.sum(weights / primary_radii)
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
        r2_vw = np.sum(weights * primary_radii**2)
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
    primary_radii=None, D0=2.0,
):
    """Save per-run comparison plot."""
    experiment_dir = os.path.join(
        base_dir, f"molecules_{int(molecules)}_timestep_{time_step}"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    plot_numerical_vs_analytical_comparison(
        difftime, D_numerical, t_analytical, D_analytical,
        molecules, time_step, run_id, experiment_dir, time_threshold,
        primary_radii=primary_radii, D0=D0,
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

def run_validation_study():
    """Run num_molecules × time_step calibration on a Gamma-distributed cylinder substrate."""
    st = time.time()

    # Physical parameters
    D0 = 2.0              # μm²/ms
    total_sim_time = 100  # ms
    time_threshold = 0.05 # ms (lower bound for MAE window)

    # Gamma distribution for axon diameters: Ŵ(κ, θ), θ in μm
    # θ = 4.5×10⁻⁷ m = 0.45 μm  →  mean diameter = κ × θ = 1.8 μm
    kappa = 4.0
    theta_um = 0.45
    n_axons = 500
    axon_area_fraction = 0.65

    # Calibration grid
    molecules_values = [int(1e6), int(5e5), int(2e5), int(1e5), int(5e4), int(2e4), int(1e4)]
    time_step_values = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    n_repeats = 10

    # Create output directory
    config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH = (
        "./tests/calibration/num_molecules_and_time_step_calibration/figs/"
        "cylinder_validation_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH, exist_ok=True)

    # ── 1. Generate substrate (once) ────────────────────────────────────────
    print("Generating Gamma-distributed cylinder substrate (L-BFGS packing)...")
    x_centers, y_centers, primary_radii, lxy = axon_gamma_dist_gen(
        n_axons=n_axons,
        axon_area_fraction=axon_area_fraction,
        kappa=kappa,
        theta_um=theta_um,
        rng_seed=42,
    )
    lx, ly, lz = lxy, lxy, 20.0
    primary_centers_xy = np.column_stack((x_centers, y_centers))

    centers_xy, radii = build_edge_ghost_cylinders(
        centers_xy=primary_centers_xy,
        radii=primary_radii,
        lx=lx,
        ly=ly,
        edge_distance=lxy / 5.0,
    )

    print(f"  Primary cylinders placed: {len(primary_radii)}")
    print(f"  Total incl. edge ghosts:  {len(radii)}")
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
    print(f"  nstructures: {sg3.nstructures}")

    # ── 3. Volume-weighted analytical D(t) (once) ───────────────────────────
    print("\nComputing volume-weighted analytical D(t)...")
    t_analytical, D_analytical = calculateDanalytical_multi_cylinder(primary_radii, D0)
    D_interp = interp1d(
        np.log10(t_analytical), D_analytical,
        bounds_error=False,
        fill_value=(D_analytical[0], D_analytical[-1]),
    )
    print("  Done.")

    # ── 4. Parameter sweep ──────────────────────────────────────────────────
    summary_results = []
    total_combinations = len(molecules_values) * len(time_step_values)
    current_combination = 0

    for molecules in molecules_values:
        for time_step in time_step_values:
            current_combination += 1
            print(f"\n{'='*40}")
            print(f"Combination {current_combination}/{total_combinations}  "
                  f"| molecules={molecules}  time_step={time_step} ms")

            mae_values = []
            computation_times = []

            for run_id in range(n_repeats):
                print(f"  Run {run_id + 1}/{n_repeats}", end=" ... ", flush=True)
                t0 = time.time()
                try:
                    difftime, D_numerical = calculateDnumerical(
                        sg3, D0, molecules, time_step, total_sim_time, run_id,
                    )
                    elapsed = time.time() - t0
                    print(f"{elapsed:.1f} s")

                    mae = calculate_mae(D_numerical, D_interp, difftime, time_threshold)
                    mae_values.append(mae)
                    computation_times.append(elapsed)

                    save_experiment_data(
                        difftime, D_numerical, t_analytical, D_analytical,
                        molecules, time_step, run_id,
                        config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH,
                        mae, time_threshold,
                        primary_radii=primary_radii, D0=D0,
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

            if mae_values:
                valid_mae = [x for x in mae_values if not np.isnan(x)]
                if valid_mae:
                    summary_results.append({
                        "molecules": molecules,
                        "time_step": time_step,
                        "time_threshold": time_threshold,
                        "mae": np.mean(valid_mae),
                        "std_mae": np.std(valid_mae),
                        "min_mae": np.min(valid_mae),
                        "max_mae": np.max(valid_mae),
                        "mean_computation_time": np.mean(computation_times),
                        "std_computation_time": np.std(computation_times),
                        "n_successful_runs": len(valid_mae),
                    })

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
    print("=======Starting Analytical Validation Study=======")
    run_validation_study()
    print("\nAnalytical validation study completed!")