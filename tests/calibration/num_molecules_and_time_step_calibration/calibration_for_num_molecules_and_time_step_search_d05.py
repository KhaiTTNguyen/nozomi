import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import hashlib
import json
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

MANIFEST_FILENAME = "experiment_config.json"
SUBSTRATE_FILENAME = "substrate_checkpoint.npz"
ANALYTICAL_FILENAME = "analytical_reference.npz"
REPEAT_RESULTS_FILENAME = "repeat_results.csv"

REPEAT_RESULT_COLUMNS = [
    "molecules",
    "time_step",
    "run_id",
    "repeat_seed",
    "time_threshold",
    "mae",
    "computation_time",
    "status",
    "started_at",
    "finished_at",
    "error",
]

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
    centers_xy, radii, lx, ly, lz, D0,
    dz_over_r=0.5, extension_fraction=0.30, min_segments=20,
):
    """Build a SimGeometry3D with z-aligned cylinders (no T2 weighting).

    Each cylinder is represented as a chain of overlapping spheres along z.
    Uses *relative* discretization: each cylinder gets a number of spheres
    chosen so that the spacing dz_i ≈ dz_over_r * r_i.  This guarantees the
    sphere chain overlaps (dz_i < 2 r_i for any dz_over_r < 2) regardless of
    the cylinder radius — important for substrates with a wide radius
    distribution (e.g. Gamma θ=0.15 μm).

    Parameters
    ----------
    dz_over_r          : target ratio of sphere spacing to cylinder radius
                         along z.  0.5 means ~2 spheres per radius along z.
    extension_fraction : periodic-extension spheres on each side of the box,
                         expressed as a fraction of n_segments_i.  0.30 matches
                         the OLD single-cylinder convention (30 of 100).
    min_segments       : floor on per-cylinder n_segments_i to keep very large
                         radii reasonably sampled.
    """
    T2 = 1e10   # effectively infinite T2 — no T2 weighting
    rho = 1.0
    sg3 = geom.SimGeometry3D(lx, ly, lz, D0, T2, rho)

    total_spheres = 0
    for (cx, cy), radius in zip(centers_xy, radii):
        n_segments_i = max(min_segments, int(np.ceil(lz / (dz_over_r * radius))))
        extension_segments_i = max(1, int(np.ceil(extension_fraction * n_segments_i)))

        base_z = np.linspace(-lz / 2.0, lz / 2.0, num=n_segments_i)
        sx = np.full_like(base_z, cx, dtype=float)
        sy = np.full_like(base_z, cy, dtype=float)
        sz = base_z
        sr = np.full_like(base_z, radius, dtype=float)

        sz_ext = np.concatenate(
            (sz[-extension_segments_i:] - lz, sz, sz[:extension_segments_i] + lz)
        )
        sx_ext = np.concatenate(
            (sx[-extension_segments_i:], sx, sx[:extension_segments_i])
        )
        sy_ext = np.concatenate(
            (sy[-extension_segments_i:], sy, sy[:extension_segments_i])
        )
        sr_ext = np.concatenate(
            (sr[-extension_segments_i:], sr, sr[:extension_segments_i])
        )

        cylinder = geom.Structure3D(sx_ext, sy_ext, sz_ext, sr_ext, D0, T2, rho)
        sg3.add_structure(cylinder)
        total_spheres += sx_ext.size

    print(f"  build_multi_cylinder_geometry: dz_over_r={dz_over_r}, "
          f"extension_fraction={extension_fraction}, total spheres={total_spheres}")
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
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "calibration_substrate_geometry.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Numerical simulation
# ---------------------------------------------------------------------------

def reseed_sim_fresh_positions(sim, struct_idxs):
    """Re-seed the simulator with fresh spin positions and a new RNG state.

    Reuses the already-compiled kernel and segment list — only the GPU spin
    positions, signal, and curand state are reset.  Used between repeats so
    each repeat starts from a different random configuration without paying
    the kernel-compile / set_segments cost again.
    """
    sim.spins = sim.geom.gpu_seed(
        sim.gpu_seed_kernel, sim.nspins, structIdxs=struct_idxs
    ).astype(np.float32)
    sim.spins_d = gpuarray.to_gpu(sim.spins)
    sim.spins0_d = sim.spins_d.copy()
    sim.sig_d.fill(np.float32(1.0))
    sim.initstates(
        np.int32(np.random.randint(np.iinfo(np.int32).max, dtype=np.int64)),
        block=(sim.nblock, 1, 1), grid=(sim.ngrid, 1),
    )


def calculateDnumerical(sim, time_step, total_sim_time=100):
    """Run DiffSim3d on an already-set-up simulator; return radial D(t).

    The caller is expected to have already created `sim`, called
    `set_segments`, `setup`, and (for repeats) `reseed_sim_fresh_positions`.
    Returns radial (transverse) D(t) = (Dx + Dy) / 2.
    """
    nt = int(total_sim_time / time_step)
    numsteps = nt + 1

    Dxarray = gpuarray.zeros(numsteps, dtype=np.float32)
    Dyarray = gpuarray.zeros(numsteps, dtype=np.float32)
    Dzarray = gpuarray.zeros(numsteps, dtype=np.float32)
    # Arrays for storing second (variance) and fourth moments (NOT used but allocated due to simulation function signature)
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
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------

def _json_ready(value):
    """Convert numpy scalars/arrays into JSON-serialisable Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _config_hash(config):
    payload = json.dumps(_json_ready(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _atomic_write_json(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _atomic_write_dataframe(df, path):
    tmp_path = f"{path}.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _summary_csv_path(base_dir, kappa, theta_um, time_threshold):
    return os.path.join(
        base_dir,
        f"cylinder_validation_summary_kappa{kappa}_theta{theta_um}"
        f"_threshold{time_threshold}.csv",
    )


def _load_manifest(base_dir):
    manifest_path = os.path.join(base_dir, MANIFEST_FILENAME)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest


def _save_manifest(base_dir, config):
    manifest = {
        "config": _json_ready(config),
        "config_hash": _config_hash(config),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_write_json(os.path.join(base_dir, MANIFEST_FILENAME), manifest)


def _save_substrate_checkpoint(base_dir, x_centers, y_centers, primary_radii,
                                primary_centers_xy, centers_xy, radii, lx, ly, lz):
    np.savez_compressed(
        os.path.join(base_dir, SUBSTRATE_FILENAME),
        x_centers=x_centers,
        y_centers=y_centers,
        primary_radii=primary_radii,
        primary_centers_xy=primary_centers_xy,
        centers_xy=centers_xy,
        radii=radii,
        lx=np.float64(lx),
        ly=np.float64(ly),
        lz=np.float64(lz),
    )


def _load_substrate_checkpoint(base_dir):
    data = np.load(os.path.join(base_dir, SUBSTRATE_FILENAME))
    return {
        "x_centers": data["x_centers"],
        "y_centers": data["y_centers"],
        "primary_radii": data["primary_radii"],
        "primary_centers_xy": data["primary_centers_xy"],
        "centers_xy": data["centers_xy"],
        "radii": data["radii"],
        "lx": float(data["lx"]),
        "ly": float(data["ly"]),
        "lz": float(data["lz"]),
    }


def _save_analytical_checkpoint(base_dir, t_analytical, D_analytical):
    np.savez_compressed(
        os.path.join(base_dir, ANALYTICAL_FILENAME),
        t_analytical=t_analytical,
        D_analytical=D_analytical,
    )


def _load_analytical_checkpoint(base_dir):
    data = np.load(os.path.join(base_dir, ANALYTICAL_FILENAME))
    return data["t_analytical"], data["D_analytical"]


def _repeat_results_path(base_dir):
    return os.path.join(base_dir, REPEAT_RESULTS_FILENAME)


def load_repeat_results(base_dir):
    path = _repeat_results_path(base_dir)
    if not os.path.exists(path):
        return pd.DataFrame(columns=REPEAT_RESULT_COLUMNS)
    df = pd.read_csv(path)
    for column in REPEAT_RESULT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[REPEAT_RESULT_COLUMNS]


def save_repeat_results(df, base_dir):
    df = df.copy()
    df = df.sort_values(["molecules", "time_step", "run_id"])
    _atomic_write_dataframe(df[REPEAT_RESULT_COLUMNS], _repeat_results_path(base_dir))


def upsert_repeat_result(base_dir, record):
    df = load_repeat_results(base_dir)
    record_df = pd.DataFrame([{column: record.get(column, np.nan) for column in REPEAT_RESULT_COLUMNS}])
    df = pd.concat([df, record_df], ignore_index=True)
    df = df.drop_duplicates(["molecules", "time_step", "run_id"], keep="last")
    save_repeat_results(df, base_dir)


def repeat_completed(repeat_results, molecules, time_step, run_id):
    if repeat_results.empty:
        return False
    mask = (
        (repeat_results["molecules"].astype(int) == int(molecules))
        & np.isclose(repeat_results["time_step"].astype(float), float(time_step))
        & (repeat_results["run_id"].astype(int) == int(run_id))
        & (repeat_results["status"] == "success")
    )
    return bool(mask.any())


def build_summary_from_repeat_results(repeat_results):
    if repeat_results.empty:
        return pd.DataFrame()
    df = repeat_results[repeat_results["status"] == "success"].copy()
    if df.empty:
        return pd.DataFrame()
    df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
    df["computation_time"] = pd.to_numeric(df["computation_time"], errors="coerce")
    df = df.dropna(subset=["mae"])
    summary = (
        df.groupby(["molecules", "time_step", "time_threshold"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            std_mae=("mae", "std"),
            min_mae=("mae", "min"),
            max_mae=("mae", "max"),
            mean_computation_time=("computation_time", "mean"),
            std_computation_time=("computation_time", "std"),
            n_successful_runs=("mae", "count"),
        )
    )
    summary[["std_mae", "std_computation_time"]] = summary[["std_mae", "std_computation_time"]].fillna(0.0)
    return summary.sort_values(["molecules", "time_step"])


def refresh_summary_checkpoint(base_dir, kappa, theta_um, time_threshold):
    repeat_results = load_repeat_results(base_dir)
    summary = build_summary_from_repeat_results(repeat_results)
    if not summary.empty:
        _atomic_write_dataframe(summary, _summary_csv_path(base_dir, kappa, theta_um, time_threshold))
    return summary


def make_repeat_seed(config_hash, molecules, time_step, run_id):
    payload = f"{config_hash}|{int(molecules)}|{float(time_step):.12g}|{int(run_id)}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def calculateDnumerical_mock(time_step, total_sim_time, D_long_interp, repeat_seed):
    """Cheap deterministic stand-in for interruption/resume testing."""
    rng = np.random.default_rng(repeat_seed)
    nt = max(1, int(total_sim_time / time_step))
    difftime = np.arange(1, nt + 1, dtype=float) * time_step
    baseline = D_long_interp(np.log10(difftime))
    noise = rng.normal(loc=0.0, scale=1e-4, size=difftime.shape)
    return difftime, np.clip(baseline + noise, 0.0, None)


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
        t_asymptotic = np.logspace(np.log10(0.035), 2, 100)
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
    df = summary_results if isinstance(summary_results, pd.DataFrame) else pd.DataFrame(summary_results)
    csv_path = _summary_csv_path(base_dir, kappa, theta_um, time_threshold)
    _atomic_write_dataframe(df, csv_path)
    print(f"\nSummary results saved to: {csv_path}")


def create_analysis_plots(
    summary_results, base_dir, kappa, theta_um, time_threshold, total_time_hours
):
    """Three-panel figure: MAE vs time step | computation time log | computation time linear."""
    df = summary_results if isinstance(summary_results, pd.DataFrame) else pd.DataFrame(summary_results)
    if df.empty:
        print("\nNo summary results available yet; skipping analysis plot.")
        return

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

def run_validation_study(resume_dir=None, mock=False, max_new_runs=None, output_dir=None,
                         dz_over_r=None, molecules_override=None,
                         time_steps_override=None, n_repeats_override=None):
    """Run num_molecules × time_step calibration with durable checkpoints."""
    st = time.time()

    default_config = {
        "D0": 2.0,
        "total_sim_time": 100,
        "time_threshold": 0.05,
        "kappa": 4.0,
        "theta_um": 0.15,
        "n_axons": 500,
        "axon_area_fraction": 0.65,
        "rng_seed": 42,
        "molecules_values": [int(1e6), int(5e5), int(2e5), int(1e5), int(5e4), int(2e4), int(1e4)],
        "time_step_values": [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01],
        "n_repeats": 5,
        "nsegx": 20,
        "nsegy": 20,
        "nsegz": 20,
        # Per-cylinder relative discretization. dz_over_r controls sphere
        # spacing along z relative to each cylinder's radius (dz_i = dz_over_r * r_i).
        # Smaller -> more spheres -> smoother cylinder wall, higher kernel cost.
        "dz_over_r": 0.5,
        "extension_fraction": 0.30,
        "min_segments": 20,
        "mock": False,
    }

    if mock:
        default_config.update({
            "total_sim_time": 0.1,
            "time_threshold": 0.01,
            "n_axons": 4,
            "molecules_values": [1000, 2000],
            "time_step_values": [0.01, 0.02],
            "n_repeats": 3,
            "mock": True,
        })

    if resume_dir is not None:
        base_dir = resume_dir
        manifest = _load_manifest(base_dir)
        config = manifest["config"]
        config_hash = manifest["config_hash"]
        print(f"Resuming checkpointed calibration from: {base_dir}")
    else:
        config = default_config
        # Apply CLI overrides BEFORE hashing so different settings get fresh dirs.
        if dz_over_r is not None:
            config["dz_over_r"] = float(dz_over_r)
        if molecules_override is not None:
            config["molecules_values"] = [int(m) for m in molecules_override]
        if time_steps_override is not None:
            config["time_step_values"] = [float(t) for t in time_steps_override]
        if n_repeats_override is not None:
            config["n_repeats"] = int(n_repeats_override)
        config_hash = _config_hash(config)
        if output_dir is None:
            suffix = "mock_" if config["mock"] else ""
            output_dir = (
                "./tests/calibration/num_molecules_and_time_step_calibration/figs/"
                f"cylinder_validation_{suffix}" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
        base_dir = output_dir
        os.makedirs(base_dir, exist_ok=True)
        _save_manifest(base_dir, config)
        save_repeat_results(pd.DataFrame(columns=REPEAT_RESULT_COLUMNS), base_dir)
        print(f"Starting new checkpointed calibration in: {base_dir}")

    config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH = base_dir
    os.makedirs(base_dir, exist_ok=True)

    D0 = float(config["D0"])
    total_sim_time = float(config["total_sim_time"])
    time_threshold = float(config["time_threshold"])
    kappa = float(config["kappa"])
    theta_um = float(config["theta_um"])
    n_axons = int(config["n_axons"])
    axon_area_fraction = float(config["axon_area_fraction"])
    rng_seed = int(config["rng_seed"])
    molecules_values = [int(value) for value in config["molecules_values"]]
    time_step_values = [float(value) for value in config["time_step_values"]]
    n_repeats = int(config["n_repeats"])
    dz_over_r = float(config.get("dz_over_r", 0.5))
    extension_fraction = float(config.get("extension_fraction", 0.30))
    min_segments = int(config.get("min_segments", 20))
    is_mock = bool(config.get("mock", False))

    # ── 1. Load or generate substrate (once) ────────────────────────────────
    substrate_path = os.path.join(base_dir, SUBSTRATE_FILENAME)
    if os.path.exists(substrate_path):
        print("Loading substrate checkpoint...")
        substrate = _load_substrate_checkpoint(base_dir)
        x_centers = substrate["x_centers"]
        y_centers = substrate["y_centers"]
        primary_radii = substrate["primary_radii"]
        primary_centers_xy = substrate["primary_centers_xy"]
        centers_xy = substrate["centers_xy"]
        radii = substrate["radii"]
        lx, ly, lz = substrate["lx"], substrate["ly"], substrate["lz"]
    elif is_mock:
        print("Generating mock substrate checkpoint...")
        primary_radii = np.array([0.20, 0.24, 0.30, 0.36], dtype=float)
        x_centers = np.array([-0.45, 0.45, -0.45, 0.45], dtype=float)
        y_centers = np.array([-0.45, -0.45, 0.45, 0.45], dtype=float)
        lx = ly = lz = 2.0
        primary_centers_xy = np.column_stack((x_centers, y_centers))
        centers_xy, radii = primary_centers_xy.copy(), primary_radii.copy()
        _save_substrate_checkpoint(
            base_dir, x_centers, y_centers, primary_radii,
            primary_centers_xy, centers_xy, radii, lx, ly, lz,
        )
    else:
        print("Generating Gamma-distributed cylinder substrate (L-BFGS packing)...")
        x_centers, y_centers, primary_radii, lxy = axon_gamma_dist_gen(
            n_axons=n_axons,
            axon_area_fraction=axon_area_fraction,
            kappa=kappa,
            theta_um=theta_um,
            rng_seed=rng_seed,
        )
        lx, ly, lz = lxy, lxy, lxy
        primary_centers_xy = np.column_stack((x_centers, y_centers))
        centers_xy, radii = build_edge_ghost_cylinders(
            centers_xy=primary_centers_xy,
            radii=primary_radii,
            lx=lx,
            ly=ly,
            edge_distance=lxy / 5.0,
        )
        _save_substrate_checkpoint(
            base_dir, x_centers, y_centers, primary_radii,
            primary_centers_xy, centers_xy, radii, lx, ly, lz,
        )

    print(f"  Primary cylinders placed: {len(primary_radii)}")
    print(f"  Total incl. edge ghosts:  {len(radii)}")
    print(f"  Box side:                 {lx:.2f} μm")
    print(f"  Mean radius (primary):    {primary_radii.mean():.3f} μm "
          f"± {primary_radii.std():.3f} μm")

    _plot_substrate_geometry(
        centers_xy=centers_xy,
        radii=radii,
        lx=lx, ly=ly,
        output_dir=base_dir,
    )

    # ── 2. Build SimGeometry3D (once) ───────────────────────────────────────
    if is_mock:
        sg3 = None
        struct_idxs = []
        print("\nMock mode: skipping SimGeometry3D/DiffSim3d setup.")
    else:
        print("\nBuilding SimGeometry3D (this may take a moment)...")
        sg3 = build_multi_cylinder_geometry(
            centers_xy, radii, lx, ly, lz, D0,
            dz_over_r=dz_over_r,
            extension_fraction=extension_fraction,
            min_segments=min_segments,
        )
        print(f"  nstructures: {sg3.nstructures}")
        struct_idxs = list(np.arange(0, sg3.nstructures))

    # ── 3. Volume-weighted analytical D(t) (once) ───────────────────────────
    analytical_path = os.path.join(base_dir, ANALYTICAL_FILENAME)
    if os.path.exists(analytical_path):
        print("\nLoading analytical D(t) checkpoint...")
        t_analytical, D_analytical = _load_analytical_checkpoint(base_dir)
    else:
        print("\nComputing volume-weighted analytical D(t)...")
        t_analytical, D_analytical = calculateDanalytical_multi_cylinder(primary_radii, D0)
        _save_analytical_checkpoint(base_dir, t_analytical, D_analytical)
    D_interp = interp1d(
        np.log10(t_analytical), D_analytical,
        bounds_error=False,
        fill_value=(D_analytical[0], D_analytical[-1]),
    )
    print("  Done.")

    # ── 4. Parameter sweep ──────────────────────────────────────────────────
    total_combinations = len(molecules_values) * len(time_step_values)
    current_combination = 0
    new_runs_completed = 0

    for molecules in molecules_values:
        repeat_results = load_repeat_results(base_dir)
        pending_for_molecules = any(
            not repeat_completed(repeat_results, molecules, time_step, run_id)
            for time_step in time_step_values
            for run_id in range(n_repeats)
        )
        if not pending_for_molecules:
            print(f"\n[Skip] All repeats already complete for molecules={molecules}.")
            current_combination += len(time_step_values)
            continue

        # Build sim ONCE per `molecules` (nspins is baked into the CUDA kernel
        # at compile time, so we only re-build when nspins changes).
        if is_mock:
            sim = None
        else:
            t_setup = time.time()
            sim = ds3.DiffSim3d(sg3, int(molecules))
            nsegx, nsegy, nsegz = int(config["nsegx"]), int(config["nsegy"]), int(config["nsegz"])
            sim.set_segments(nsegx=nsegx, nsegy=nsegy, nsegz=nsegz)
            print(f"\n[Setup] Building DiffSim3d for molecules={molecules}, segments=({nsegx}, {nsegy}, {nsegz}) "
                  f"...")
            sim.setup(structures=struct_idxs)
            print(f"[Setup] done in {time.time() - t_setup:.1f} s  "
                  f"(nspheres_per_seg={sim.spheres_per_segment})")

        for time_step in time_step_values:
            current_combination += 1
            print(f"\n{'='*40}")
            print(f"Combination {current_combination}/{total_combinations}  "
                  f"| molecules={molecules}  time_step={time_step} ms")

            for run_id in range(n_repeats):
                repeat_results = load_repeat_results(base_dir)
                if repeat_completed(repeat_results, molecules, time_step, run_id):
                    print(f"  Run {run_id + 1}/{n_repeats} already complete; skipping.")
                    continue

                if max_new_runs is not None and new_runs_completed >= max_new_runs:
                    print(f"\nReached --max-new-runs={max_new_runs}; stopping early for interruption test.")
                    summary = refresh_summary_checkpoint(base_dir, kappa, theta_um, time_threshold)
                    total_hours = (time.time() - st) / 3600.0
                    create_analysis_plots(summary, base_dir, kappa, theta_um, time_threshold, total_hours)
                    return base_dir

                repeat_seed = make_repeat_seed(config_hash, molecules, time_step, run_id)
                print(f"  Run {run_id + 1}/{n_repeats}", end=" ... ", flush=True)
                t0 = time.time()
                started_at = datetime.now().isoformat(timespec="seconds")
                upsert_repeat_result(base_dir, {
                    "molecules": int(molecules),
                    "time_step": float(time_step),
                    "run_id": int(run_id),
                    "repeat_seed": int(repeat_seed),
                    "time_threshold": float(time_threshold),
                    "status": "running",
                    "started_at": started_at,
                    "finished_at": "",
                    "error": "",
                })
                try:
                    if is_mock:
                        difftime, D_numerical = calculateDnumerical_mock(
                            time_step, total_sim_time, D_interp, repeat_seed,
                        )
                    else:
                        np.random.seed(repeat_seed)
                        reseed_sim_fresh_positions(sim, struct_idxs)
                        difftime, D_numerical = calculateDnumerical(
                            sim, time_step, total_sim_time,
                        )
                    elapsed = time.time() - t0
                    print(f"{elapsed:.1f} s")

                    mae = calculate_mae(D_numerical, D_interp, difftime, time_threshold)

                    save_experiment_data(
                        difftime, D_numerical, t_analytical, D_analytical,
                        molecules, time_step, run_id,
                        base_dir,
                        mae, time_threshold,
                        primary_radii=primary_radii, D0=D0,
                    )
                    upsert_repeat_result(base_dir, {
                        "molecules": int(molecules),
                        "time_step": float(time_step),
                        "run_id": int(run_id),
                        "repeat_seed": int(repeat_seed),
                        "time_threshold": float(time_threshold),
                        "mae": float(mae),
                        "computation_time": float(elapsed),
                        "status": "success",
                        "started_at": started_at,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                        "error": "",
                    })
                    new_runs_completed += 1
                except Exception as e:
                    print(f"ERROR: {e}")
                    upsert_repeat_result(base_dir, {
                        "molecules": int(molecules),
                        "time_step": float(time_step),
                        "run_id": int(run_id),
                        "repeat_seed": int(repeat_seed),
                        "time_threshold": float(time_threshold),
                        "status": "failed",
                        "started_at": started_at,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                        "error": str(e),
                    })
                    continue

            refresh_summary_checkpoint(base_dir, kappa, theta_um, time_threshold)

    total_hours = (time.time() - st) / 3600.0
    summary_results = refresh_summary_checkpoint(base_dir, kappa, theta_um, time_threshold)
    save_summary_results(
        summary_results,
        base_dir,
        kappa, theta_um, time_threshold,
    )
    create_analysis_plots(
        summary_results,
        base_dir,
        kappa, theta_um, time_threshold, total_hours,
    )
    return base_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run or resume the num_molecules/time_step calibration study."
    )
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Existing calibration output directory to resume from.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a tiny synthetic calibration grid for checkpoint/resume testing.",
    )
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=None,
        help="Stop after this many newly completed repeats; useful for interruption tests.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for a new run. Ignored when --resume-dir is used.",
    )
    parser.add_argument(
        "--dz-over-r",
        type=float,
        default=None,
        help="Override dz_over_r (per-cylinder relative discretization).",
    )
    parser.add_argument(
        "--molecules",
        type=int,
        nargs="+",
        default=None,
        help="Override molecules_values list (e.g. --molecules 200000).",
    )
    parser.add_argument(
        "--time-steps",
        type=float,
        nargs="+",
        default=None,
        help="Override time_step_values list (e.g. --time-steps 0.001).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Override n_repeats.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=======Starting Analytical Validation Study=======")
    run_validation_study(
        resume_dir=args.resume_dir,
        mock=args.mock,
        max_new_runs=args.max_new_runs,
        output_dir=args.output_dir,
        dz_over_r=args.dz_over_r,
        molecules_override=args.molecules,
        time_steps_override=args.time_steps,
        n_repeats_override=args.n_repeats,
    )
    print("\nAnalytical validation study completed!")