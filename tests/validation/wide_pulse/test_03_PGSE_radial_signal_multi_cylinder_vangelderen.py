import pycuda.autoinit
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
from scipy.special import jnp_zeros
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from simulation_toolkit.simulation_engine.diffsim3d import DwiSim3d
from simulation_toolkit.simulation_engine.geometry import SimGeometry3D, Structure3D
from simulation_toolkit.simulation_engine.waveforms import PGDiffWaveform
from simulation_toolkit.simulation_engine.helper import plot_gradient_experiment as plt_ge

INCLUDE_SIMULATED_DWI_VALUE_LABELS = True


def van_gelderen_radial_signal(big_delta, little_delta, gmax, radius, diffusivity, n_roots=200):
    """Analytical radial signal for a single cylinder (van Gelderen Eq. 11)."""
    gamma = 0.0267513  # rad/(G*ms)
    # Simulator scales gradients in mT/m; van Gelderen Eq. 11 uses G/cm.
    # 1 mT/m = 0.01 G/cm.
    gmax_gc = np.asarray(gmax, dtype=float) * 1e-2

    alpha_r = jnp_zeros(1, n_roots)
    alpha = alpha_r / radius

    exp1 = np.exp(-diffusivity * alpha**2 * little_delta)
    exp2 = np.exp(-diffusivity * alpha**2 * big_delta)
    exp3 = np.exp(-diffusivity * alpha**2 * (big_delta - little_delta))
    exp4 = np.exp(-diffusivity * alpha**2 * (big_delta + little_delta))

    numerator = (
        2 * diffusivity * alpha**2 * little_delta
        - 2
        + 2 * exp1
        + 2 * exp2
        - exp3
        - exp4
    )
    denominator = alpha**6 * (radius**2 * alpha**2 - 1)
    series_sum = np.sum(numerator / denominator)

    # Match MATLAB vanGelderen.m:
    # sig = exp(-2*gamma^2*Gmax.^2/D.^2*sumS)
    return np.exp(-2 * gamma**2 * (gmax_gc**2) * series_sum / (diffusivity**2))


def van_gelderen_multi_cylinder_signal(big_delta, little_delta, gmax, radii, diffusivity):
    """Volume-weighted van Gelderen signal for a set of parallel cylinders.

    For intra-axonal signal with no exchange, the bundle signal is the
    weighted sum of per-cylinder signals. With equal cylinder lengths and
    identical density, weights are proportional to cross-sectional area.
    """
    radii = np.asarray(radii, dtype=float)
    weights = radii**2
    weights = weights / np.sum(weights)

    signal = np.zeros_like(gmax, dtype=float)
    for w, radius in zip(weights, radii):
        signal += w * van_gelderen_radial_signal(
            big_delta=big_delta,
            little_delta=little_delta,
            gmax=gmax,
            radius=radius,
            diffusivity=diffusivity,
        )
    return signal


def axon_delta_gen(n_axons, axon_area_fraction, mean_diameter_um, rng_seed=0):
    """Generate primary axon packing in a 2D periodic square.

    Returns one center per accepted axon, centered in ``[-L/2, L/2]``, the
    outer radii, and box size ``L``.
    """
    if not (0.0 < axon_area_fraction < 1.0):
        raise ValueError("axon_area_fraction must be in (0, 1)")

    axon_diameters = np.full(n_axons, float(mean_diameter_um), dtype=float)
    area = np.sum(np.pi * (axon_diameters / 2.0) ** 2) / float(axon_area_fraction)
    l_box = float(np.sqrt(area))

    axon_diameters = np.sort(axon_diameters)[::-1]
    x0 = []
    y0 = []
    d0 = []

    rng = np.random.default_rng(rng_seed)

    for diameter in axon_diameters:
        while True:
            xi = rng.random() * l_box
            yi = rng.random() * l_box

            if len(x0) == 0:
                break

            x_arr = np.asarray(x0)
            y_arr = np.asarray(y0)
            d_arr = np.asarray(d0)
            dx = np.abs(xi - x_arr)
            dy = np.abs(yi - y_arr)
            dx = np.minimum(dx, l_box - dx)
            dy = np.minimum(dy, l_box - dy)
            sep = np.sqrt(dx**2 + dy**2)
            min_sep = diameter / 2.0 + d_arr / 2.0
            if np.all(sep > min_sep):
                break

        # Store in [0, L] so the minimum-image distance check above stays
        # in a consistent coordinate frame throughout the loop.
        x0.append(xi)
        y0.append(yi)
        d0.append(diameter)

    # Shift to [-L/2, L/2] only after all placements are done.
    x0 = np.asarray(x0, dtype=float) - l_box / 2.0
    y0 = np.asarray(y0, dtype=float) - l_box / 2.0
    r0 = np.asarray(d0, dtype=float) / 2.0
    return x0, y0, r0, l_box


def build_edge_ghost_cylinders(centers_xy, radii, lx, ly, edge_distance):
    """Add periodic ghost cylinders only near x/y boundaries.

    Each primary cylinder is kept once. Additional copies are added when the
    cylinder center lies within ``edge_distance`` of an x or y boundary.
    Corner copies are included when both x and y conditions are met.
    """
    x_min = -lx / 2.0
    x_max = lx / 2.0
    y_min = -ly / 2.0
    y_max = ly / 2.0

    expanded_centers = []
    expanded_radii = []

    for (cx, cy), radius in zip(np.asarray(centers_xy, dtype=float), np.asarray(radii, dtype=float)):
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

    return np.asarray(expanded_centers, dtype=float), np.asarray(expanded_radii, dtype=float)


def _build_multi_periodic_cylinder_geometry(
    lx,
    ly,
    lz,
    centers_xy,
    radii,
    diffusivity,
    t2,
    rho,
    n_segments=200,
    extension_segments=30,
):
    """Build multiple z-aligned cylinders as separate structures."""
    sg3 = SimGeometry3D(lx, ly, lz, diffusivity, t2, rho)

    base_z = np.linspace(-lz / 2, lz / 2, num=n_segments)

    for (cx, cy), radius in zip(centers_xy, radii):
        sx = np.full_like(base_z, cx, dtype=float)
        sy = np.full_like(base_z, cy, dtype=float)
        sz = base_z.copy()
        sr = np.full_like(base_z, radius, dtype=float)

        sz_ext = np.concatenate((sz[-extension_segments:] - lz, sz, sz[:extension_segments] + lz), axis=0)
        sx_ext = np.concatenate((sx[-extension_segments:], sx, sx[:extension_segments]), axis=0)
        sy_ext = np.concatenate((sy[-extension_segments:], sy, sy[:extension_segments]), axis=0)
        sr_ext = np.concatenate((sr[-extension_segments:], sr, sr[:extension_segments]), axis=0)

        cylinder = Structure3D(sx_ext, sy_ext, sz_ext, sr_ext, diffusivity, t2, rho)
        sg3.add_structure(cylinder)

    return sg3


def _plot_axon_geometry(centers_xy, radii, lx, ly, output_dir):
    """Plot axon cross-section geometry used by test03 and save to disk."""
    x = centers_xy[:, 0]
    y = centers_xy[:, 1]
    r = np.asarray(radii)

    in_box = (
        (x >= -lx / 2.0)
        & (x <= lx / 2.0)
        & (y >= -ly / 2.0)
        & (y <= ly / 2.0)
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    for xi, yi, ri in zip(x[~in_box], y[~in_box], r[~in_box]):
        ax.add_patch(
            Circle((xi, yi), ri, fill=False, linewidth=0.6, edgecolor="tab:gray", alpha=0.35)
        )

    for xi, yi, ri in zip(x[in_box], y[in_box], r[in_box]):
        ax.add_patch(
            Circle((xi, yi), ri, fill=False, linewidth=1.0, edgecolor="tab:blue", alpha=0.95)
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.1 * lx, 1.1 * lx)
    ax.set_ylim(-1.1 * ly, 1.1 * ly)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Test03 axon geometry (blue: primary box, gray: periodic copies)")

    # Draw primary simulation box boundary.
    ax.plot(
        [-lx / 2.0, lx / 2.0, lx / 2.0, -lx / 2.0, -lx / 2.0],
        [-ly / 2.0, -ly / 2.0, ly / 2.0, ly / 2.0, -ly / 2.0],
        "k-",
        linewidth=1.2,
    )

    ax.text(
        0.02,
        0.02,
        f"Primary/ghost cylinders in box: {np.count_nonzero(in_box)}\nTotal cylinders (incl. edge ghosts): {centers_xy.shape[0]}",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_03_axon_geometry.png"), dpi=150)
    plt.close(fig)


def _run_radial_diffusion_multi_cylinder_van_gelderen_pgse_case(
    mean_axon_diameter_um,
    case_label,
    plot_dir,
    include_simulated_dwi_value_labels=False,
):
    """Run one radial PGSE multi-cylinder validation case.

    Parameters:
    mean_axon_diameter_um: Mean axon diameter used by geometry generator.
    case_label: Short label for plot titles and output file names.
    """
    big_delta = 20.0
    little_delta = 5.0
    te = 30.0
    time_step = 0.01

    # Compartment settings for intra-axonal validation
    diffusivity = 3.0  # um^2/ms (non-myelin / intra-axonal in this test)
    t2 = 1e10  # ms (effectively no T2 decay)
    rho = 1.0

    # Match MATLAB-like geometry generation
    n_axons = 200
    axon_area_fraction = 0.5
    g_ratio = 0.98
    x_centers, y_centers, outer_radii, lxy = axon_delta_gen(
        n_axons=n_axons,
        axon_area_fraction=axon_area_fraction,
        mean_diameter_um=mean_axon_diameter_um,
        rng_seed=0,
    )
    inner_radii = outer_radii * g_ratio

    # Simulation settings
    spins = int(100000)
    lx, ly, lz = lxy, lxy, 20.0

    # Build cylinders from inner radii to emulate the intra-axonal domain.
    primary_centers_xy = np.column_stack((x_centers, y_centers))
    primary_radii = inner_radii
    ghost_edge_distance = lxy / 5.0
    centers_xy, radii = build_edge_ghost_cylinders(
        centers_xy=primary_centers_xy,
        radii=primary_radii,
        lx=lx,
        ly=ly,
        edge_distance=ghost_edge_distance,
    )

    sg3 = _build_multi_periodic_cylinder_geometry(
        lx=lx,
        ly=ly,
        lz=lz,
        centers_xy=centers_xy,
        radii=radii,
        diffusivity=diffusivity,
        t2=t2,
        rho=rho,
    )

    # Diffusion directions:
    # - radial: x-axis (perpendicular to cylinder axis z)
    # - axial:  z-axis (parallel to cylinder axis)
    diffdir = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ]
    )

    sim = DwiSim3d(sg3, spins)
    sim.set_diffusion_directions(diffdir=diffdir)
    sim.set_segments(nsegx=5, nsegy=5, nsegz=5)

    # Seed spins inside all cylinder structures (intra-only signal).
    sim.setup(structures=list(np.arange(0, sg3.nstructures)))

    gwave = PGDiffWaveform(
        big_delta=big_delta,
        little_delta=little_delta,
        te=te,
        time_step=time_step,
    )
    # rows: diffusion directions (x then z), columns: all spins
    phase_sig_all_dirs = sim.simulate_multi_directions(gwave)
    phase_sig_radial = phase_sig_all_dirs[0, :]
    phase_sig_axial = phase_sig_all_dirs[1, :]

    # Requested fixed b-values (s/mm^2): 0, 500, 1000, 1500, 2000, 2500, 3000.
    bval_s_mm2 = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0], dtype=float)
    bval_ms_um2 = bval_s_mm2 / 1000.0

    b_base = gwave.calculate_bvalue_from_wave()
    gmax = np.sqrt(bval_ms_um2 / b_base)

    sim_signal = np.zeros_like(bval_ms_um2)
    sim_signal_axial = np.zeros_like(bval_ms_um2)
    for idx, g in enumerate(gmax):
        sim_signal[idx] = np.mean(np.cos(g * phase_sig_radial))
        sim_signal_axial[idx] = np.mean(np.cos(g * phase_sig_axial))

    ana_signal = van_gelderen_multi_cylinder_signal(
        big_delta=big_delta,
        little_delta=little_delta,
        gmax=gmax,
        radii=primary_radii,
        diffusivity=diffusivity,
    )

    residual = sim_signal - ana_signal
    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(np.abs(residual))
    ana_abs = np.abs(ana_signal)
    valid_mape = ana_abs > 1e-12
    if np.any(valid_mape):
        mape = np.mean(np.abs(residual[valid_mape] / ana_signal[valid_mape])) * 100.0
    else:
        mape = np.nan

    # Along z-axis, intra-axonal diffusion is free in this geometry.
    ana_signal_axial = np.exp(-bval_ms_um2 * diffusivity)
    residual_axial = sim_signal_axial - ana_signal_axial
    rmse_axial = np.sqrt(np.mean(residual_axial ** 2))
    mae_axial = np.mean(np.abs(residual_axial))
    ana_abs_axial = np.abs(ana_signal_axial)
    valid_mape_axial = ana_abs_axial > 1e-12
    if np.any(valid_mape_axial):
        mape_axial = np.mean(
            np.abs(residual_axial[valid_mape_axial] / ana_signal_axial[valid_mape_axial])
        ) * 100.0
    else:
        mape_axial = np.nan

    os.makedirs(plot_dir, exist_ok=True)

    # Reuse the same plotting utility used by test_01.
    plt_ge.plot_wave(gwave, plot_dir)
    _plot_axon_geometry(centers_xy=centers_xy, radii=radii, lx=lx, ly=ly, output_dir=plot_dir)

    fig, ax_signal = plt.subplots()

    ax_signal.plot(bval_s_mm2, sim_signal, "o", label="Monte Carlo (multi-cylinder radial)")
    if include_simulated_dwi_value_labels:
        for bx, sy in zip(bval_s_mm2, sim_signal):
            ax_signal.annotate(
                f"{sy:.3f}",
                (bx, sy),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color="tab:blue",
            )
    ax_signal.plot(bval_s_mm2, ana_signal, "-", linewidth=2, label="van Gelderen weighted analytical")
    ax_signal.grid(True, linestyle="--", alpha=0.5)
    ax_signal.set_xlabel("b-value (s/mm²)")
    ax_signal.set_ylabel("DWI signal")
    ax_signal.set_title(
        f"PGSE radial signal ({case_label} um): simulation vs van Gelderen (0-3000)"
    )
    metrics_text = (
        f"RMSE = {rmse:.6f}\n"
        f"MAE = {mae:.6f}\n"
        f"MAPE = {mape:.3f}%"
    )
    ax_signal.text(
        0.03,
        0.33,
        metrics_text,
        transform=ax_signal.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )

    ax_signal.plot(
        bval_s_mm2,
        residual,
        "s",
        color="tab:green",
        label="Residual: simulated - analytical",
    )
    ax_signal.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_signal.legend(loc="best")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plot_dir,
            f"test_03_pgse_multi_cylinder_radial_vs_vangelderen_{case_label}um_0_to_3000.png",
        ),
        dpi=150,
    )
    plt.close(fig)

    fig, ax_axial = plt.subplots()
    bval_s_mm2_smooth = np.linspace(0.0, float(np.max(bval_s_mm2)), 300)
    bval_ms_um2_smooth = bval_s_mm2_smooth / 1000.0
    ana_signal_axial_smooth = np.exp(-bval_ms_um2_smooth * diffusivity)

    ax_axial.plot(
        bval_s_mm2,
        sim_signal_axial,
        "o",
        label="Monte Carlo (multi-cylinder axial)",
    )
    if include_simulated_dwi_value_labels:
        for bx, sy in zip(bval_s_mm2, sim_signal_axial):
            ax_axial.annotate(
                f"{sy:.3f}",
                (bx, sy),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color="tab:blue",
            )
    ax_axial.plot(
        bval_s_mm2_smooth,
        ana_signal_axial_smooth,
        "-",
        linewidth=2,
        label="Analytical free diffusion: exp(-bD)",
    )
    ax_axial.plot(
        bval_s_mm2,
        residual_axial,
        "s",
        color="tab:green",
        label="Residual: simulated - analytical",
    )
    ax_axial.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_axial.grid(True, linestyle="--", alpha=0.5)
    ax_axial.set_xlabel("b-value (s/mm²)")
    ax_axial.set_ylabel("DWI signal")
    ax_axial.set_title(
        f"PGSE axial signal ({case_label} um): simulation vs free diffusion (0-3000)"
    )
    metrics_text_axial = (
        f"RMSE = {rmse_axial:.6f}\n"
        f"MAE = {mae_axial:.6f}\n"
        f"MAPE = {mape_axial:.3f}%"
    )
    ax_axial.text(
        0.03,
        0.33,
        metrics_text_axial,
        transform=ax_axial.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )
    ax_axial.legend(loc="best")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plot_dir,
            f"test_03_pgse_multi_cylinder_axial_vs_free_diffusion_{case_label}um_0_to_3000.png",
        ),
        dpi=150,
    )
    plt.close(fig)

    # For regression safety, require stable agreement.
    assert rmse < 0.02, f"RMSE too high: {rmse:.6f}"
    assert rmse_axial < 0.02, f"Axial RMSE too high: {rmse_axial:.6f}"


def test_radial_diffusion_signal_perpendicular_to_multi_cylinder_van_gelderen_pgse():
    """Validation case with mean axon diameter = 12.0 um."""
    output_folder = "./tests/validation/wide_pulse/test_figures"
    folder_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_dir = os.path.join(output_folder, folder_date_time)

    _run_radial_diffusion_multi_cylinder_van_gelderen_pgse_case(
        mean_axon_diameter_um=12.0,
        case_label="12p00",
        plot_dir=plot_dir,
        include_simulated_dwi_value_labels=INCLUDE_SIMULATED_DWI_VALUE_LABELS,
    )

    """Validation case with mean axon diameter = 1.96 um."""
    _run_radial_diffusion_multi_cylinder_van_gelderen_pgse_case(
        mean_axon_diameter_um=1.96,
        case_label="1p96",
        plot_dir=plot_dir,
        include_simulated_dwi_value_labels=INCLUDE_SIMULATED_DWI_VALUE_LABELS,
    )


if __name__ == "__main__":
    test_radial_diffusion_signal_perpendicular_to_multi_cylinder_van_gelderen_pgse()
