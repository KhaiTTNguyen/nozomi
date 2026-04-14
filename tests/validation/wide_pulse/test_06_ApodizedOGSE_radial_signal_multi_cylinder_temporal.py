import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
from scipy.special import jnp_zeros
import matplotlib.pyplot as plt

import pycuda.autoinit

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from simulation_toolkit.simulation_engine.diffsim3d import DwiSim3d
from simulation_toolkit.simulation_engine.geometry import SimGeometry3D, Structure3D
from simulation_toolkit.simulation_engine.waveforms import ApodizedCosineOGSEWaveform


def _reposition_apodized_waveform_blocks(waveform, first_start_ms, inter_block_gap_ms):
    """Reposition apodized OGSE blocks to explicit timing: block1, gap, block2."""
    t_duration_ms = waveform.T_duration
    second_start_ms = first_start_ms + t_duration_ms + inter_block_gap_ms
    second_end_ms = second_start_ms + t_duration_ms
    if second_end_ms > waveform.te + 1e-9:
        raise ValueError(
            "Requested OGSE timing exceeds TE: "
            f"second_end={second_end_ms:.3f} ms, TE={waveform.te:.3f} ms"
        )

    info = waveform.get_waveform_info()
    apod_duration = info["apodization_duration_ms"]
    cosine_duration = info["cosine_duration_ms"]
    f_cosine = info["cosine_frequency_Hz"] / 1000.0

    waveform.wave[:] = 0.0
    waveform._generate_single_waveform(first_start_ms, 1.0, apod_duration, cosine_duration, f_cosine)
    waveform._generate_single_waveform(second_start_ms, -1.0, apod_duration, cosine_duration, f_cosine)


def _plot_used_ogse_waveforms(waveform_records, output_dir):
    """Plot OGSE waveforms used in test06 as separate figures."""
    for rec in waveform_records:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            rec["t_ms"],
            rec["wave_mT_m"],
            linewidth=1.5,
            color="tab:blue",
        )
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("gradient (mT/m)")
        ax.set_title(
            f"Test06 waveform: target {rec['f_target_hz']:.0f} Hz (actual {rec['f_actual_hz']:.0f} Hz)"
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f"test_06_apodized_ogse_waveform_target_{rec['f_target_hz']:.0f}Hz_actual_{rec['f_actual_hz']:.0f}Hz.png",
            ),
            dpi=150,
        )
        plt.close(fig)


def _build_grid_cylinder_geometry(
    lx,
    ly,
    lz,
    diffusivity,
    t2,
    rho,
    nx=5,
    ny=5,
    diameter_um=1.96,
    spacing_um=2.1,
    n_segments=200,
    extension_segments=30,
):
    """Build a 5x5 grid of z-aligned cylinders."""
    sg3 = SimGeometry3D(lx, ly, lz, diffusivity, t2, rho)

    radius = 0.5 * float(diameter_um)
    x_coords = (np.arange(nx, dtype=float) - (nx - 1) / 2.0) * float(spacing_um)
    y_coords = (np.arange(ny, dtype=float) - (ny - 1) / 2.0) * float(spacing_um)

    base_z = np.linspace(-lz / 2.0, lz / 2.0, num=n_segments)
    for cx in x_coords:
        for cy in y_coords:
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


def _plot_cylinder_grid_geometry_3d(output_dir, lx, ly, lz, nx, ny, spacing_um, radius_um):
    """Save a publication-style 3D rendering of the full cylinder grid geometry."""
    x_coords = (np.arange(nx, dtype=float) - (nx - 1) / 2.0) * float(spacing_um)
    y_coords = (np.arange(ny, dtype=float) - (ny - 1) / 2.0) * float(spacing_um)

    n_theta = 90
    n_z = 80
    n_r = 40
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    z_lin = np.linspace(-lz / 2.0, lz / 2.0, n_z)
    theta_side, z_side = np.meshgrid(theta, z_lin)
    r_lin = np.linspace(0.0, radius_um, n_r)
    theta_cap, r_cap = np.meshgrid(theta, r_lin)

    z_top = lz / 2.0
    z_bottom = -lz / 2.0

    fig = plt.figure(figsize=(9.2, 7.8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    side_color = "#5DA5DA"
    cap_color = "#2C7FB8"
    edge_color = "#1D3557"

    for cx in x_coords:
        for cy in y_coords:
            x_side = cx + radius_um * np.cos(theta_side)
            y_side = cy + radius_um * np.sin(theta_side)

            x_cap = cx + r_cap * np.cos(theta_cap)
            y_cap = cy + r_cap * np.sin(theta_cap)
            z_cap_top = np.full_like(x_cap, z_top)
            z_cap_bottom = np.full_like(x_cap, z_bottom)

            ax.plot_surface(
                x_side,
                y_side,
                z_side,
                rstride=1,
                cstride=1,
                color=side_color,
                edgecolor="none",
                linewidth=0.0,
                antialiased=True,
                alpha=0.96,
                shade=True,
            )
            ax.plot_surface(
                x_cap,
                y_cap,
                z_cap_top,
                rstride=1,
                cstride=1,
                color=cap_color,
                edgecolor="none",
                linewidth=0.0,
                antialiased=True,
                alpha=0.98,
                shade=True,
            )
            ax.plot_surface(
                x_cap,
                y_cap,
                z_cap_bottom,
                rstride=1,
                cstride=1,
                color=cap_color,
                edgecolor="none",
                linewidth=0.0,
                antialiased=True,
                alpha=0.98,
                shade=True,
            )

            # Subtle top and bottom rims to improve perceived sharpness in print.
            ax.plot(
                cx + radius_um * np.cos(theta),
                cy + radius_um * np.sin(theta),
                np.full_like(theta, z_top),
                color=edge_color,
                linewidth=0.5,
                alpha=0.65,
            )
            ax.plot(
                cx + radius_um * np.cos(theta),
                cy + radius_um * np.sin(theta),
                np.full_like(theta, z_bottom),
                color=edge_color,
                linewidth=0.4,
                alpha=0.45,
            )

    ax.set_xlim(-lx / 2.0, lx / 2.0)
    ax.set_ylim(-ly / 2.0, ly / 2.0)
    ax.set_zlim(-lz / 2.0, lz / 2.0)
    ax.set_box_aspect((lx, ly, lz))
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_zlabel("z (um)")
    ax.set_title(
        f"Test06 geometry: {nx}x{ny} cylinders (diameter={2.0 * radius_um:.2f} um, spacing={spacing_um:.2f} um)",
        pad=14,
    )
    ax.view_init(elev=26, azim=40)
    ax.grid(False)

    # Keep axes clean and publication-friendly.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "test_06_cylinder_geometry_3d.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _cosine_ogse_signal_eq11(
    gmax,
    frequency_hz,
    sigma_ms,
    tau_ms,
    radius_um,
    diffusivity,
    n_roots=120,
    beta=None,
):
    """Analytical OGSE signal using Eq. (10)+(11) style attenuation."""
    if beta is None:
        beta = jnp_zeros(1, n_roots)

    g_arr = np.asarray(gmax, dtype=float)
    gamma = 267.513  # rad/(ms*mT)
    g_mT_um = g_arr * 1e-6  # mT/m -> mT/um

    omega = 2.0 * np.pi * float(frequency_hz) / 1000.0  # rad/ms
    mu_n = np.asarray(beta, dtype=float)
    b_n = 2.0 * (radius_um / mu_n) ** 2 / (mu_n**2 - 1.0)
    lambda_n = (mu_n / radius_um) ** 2  # 1/um^2

    lam_d = lambda_n * diffusivity
    lam2_d2 = (lambda_n**2) * (diffusivity**2)

    if abs(omega) < 1e-14:
        trig_term = sigma_ms
    else:
        trig_term = sigma_ms / 2.0 + np.sin(2.0 * omega * sigma_ms) / (4.0 * omega)

    a = lam_d * tau_ms
    b = lam_d * sigma_ms
    exp_term = np.exp(-a) - 0.5 * np.exp(-(a - b)) - 0.5 * np.exp(-(a + b))

    bracket = (
        ((lam2_d2 + omega**2) * trig_term) / lam_d
        - 1.0
        + np.exp(-b)
        + exp_term
    )

    denom = (lam2_d2 + omega**2) ** 2
    modal_contrib = (b_n * lam2_d2 / denom) * bracket
    modal_contrib = np.nan_to_num(modal_contrib, nan=0.0, posinf=0.0, neginf=0.0)
    modal_sum = np.sum(modal_contrib)

    beta_2tau = 2.0 * (gamma * g_mT_um) ** 2 * modal_sum
    return np.exp(-beta_2tau)


def _normalize_spin_array(spins_xyz):
    """Return spin positions as an array shaped (nspins, 3)."""
    spins_xyz = np.asarray(spins_xyz, dtype=float)
    if spins_xyz.ndim != 2:
        raise ValueError("spins_xyz must be a 2D array.")

    if spins_xyz.shape[1] == 3:
        return spins_xyz
    if spins_xyz.shape[0] == 3:
        return spins_xyz.T

    if spins_xyz.shape[1] > 3:
        return spins_xyz[:, :3]
    if spins_xyz.shape[0] > 3:
        return spins_xyz[:3, :].T

    raise ValueError("spins_xyz must contain x, y, z coordinates.")


def _plot_final_spin_positions_3d_all(spins_xyz, lx, ly, lz, output_dir):
    """Save a 3D scatter of all final spin positions."""
    spins_xyz = _normalize_spin_array(spins_xyz)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        spins_xyz[:, 0],
        spins_xyz[:, 1],
        spins_xyz[:, 2],
        s=2,
        alpha=0.18,
        color="tab:blue",
        depthshade=False,
    )

    ax.set_xlim(-lx / 2.0, lx / 2.0)
    ax.set_ylim(-ly / 2.0, ly / 2.0)
    ax.set_zlim(-lz / 2.0, lz / 2.0)
    ax.set_box_aspect((lx, ly, lz))
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_zlabel("z (um)")
    ax.set_title("Test06: Final spin positions (3D, all spins) after apodized OGSE simulation")
    ax.view_init(elev=24, azim=42)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_06_final_spin_positions_3d_all.png"), dpi=170)
    plt.close(fig)


def test_radial_diffusion_signal_perpendicular_multi_cylinder_apodized_ogse_temporal_grid_5x5():
    """Test06: apodized OGSE setup with a 5x5 cylinder grid geometry."""
    diffusivity = 1.0  # um^2/ms
    diameter_um = 1.96
    radius_um = diameter_um / 2.0
    spacing_um = 2.1

    frequency_targets_hz = [100.0, 250.0, 500.0, 1000.0]
    t_duration_ms = 20.0
    te_ms = 2.0 * t_duration_ms
    first_block_start_ms = 0.0
    inter_block_gap_ms = 0.0

    output_folder = "./tests/validation/wide_pulse/test_figures"
    folder_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_dir = os.path.join(output_folder, folder_date_time)
    os.makedirs(plot_dir, exist_ok=True)

    lx, ly, lz = 20.0, 20.0, 20.0
    sg3 = _build_grid_cylinder_geometry(
        lx=lx,
        ly=ly,
        lz=lz,
        diffusivity=diffusivity,
        t2=200,
        rho=1.0,
        nx=5,
        ny=5,
        diameter_um=diameter_um,
        spacing_um=spacing_um,
    )
    _plot_cylinder_grid_geometry_3d(
        output_dir=plot_dir,
        lx=lx,
        ly=ly,
        lz=lz,
        nx=5,
        ny=5,
        spacing_um=spacing_um,
        radius_um=radius_um,
    )

    num_spins = 50000
    sim = DwiSim3d(sg3, num_spins)
    sim.set_diffusion_directions(
        diffdir=np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
            ]
        )
    )
    sim.set_segments(nsegx=20, nsegy=20, nsegz=20)
    sim.setup(structures=list(np.arange(0, sg3.nstructures)))

    n_roots = 200
    beta_roots = jnp_zeros(1, n_roots)

    bval_s_mm2 = np.linspace(0.0, 500.0, 11)
    bval_s_mm2_extended = np.concatenate(
        [
            bval_s_mm2,
            np.array([1000.0, 1500.0, 2000.0, 2500.0, 3000.0], dtype=float),
        ]
    )

    waveform_records = []
    fig_signal, ax_signal = plt.subplots(figsize=(9, 6))
    fig_axial, ax_axial = plt.subplots(figsize=(9, 6))
    fig_radial_extended, ax_radial_extended = plt.subplots(figsize=(9, 6))
    summary_lines = ["Test06 apodized OGSE Monte Carlo vs analytical signal summary", "=" * 61]

    for f_target_hz in frequency_targets_hz:
        sim.reset_simulation()
        n_cycles = int(round(f_target_hz * t_duration_ms / 1000.0))
        if n_cycles <= 0:
            raise ValueError("Computed n_cycles must be positive.")

        waveform = ApodizedCosineOGSEWaveform(
            N_cycles=n_cycles,
            T_duration=t_duration_ms,
            te=te_ms,
            gmax=1.0,
            time_step=0.01,
        )
        _reposition_apodized_waveform_blocks(
            waveform=waveform,
            first_start_ms=first_block_start_ms,
            inter_block_gap_ms=inter_block_gap_ms,
        )

        info = waveform.get_waveform_info()
        f_hz = info["cosine_frequency_Hz"]
        waveform_records.append(
            {
                "f_target_hz": float(f_target_hz),
                "f_actual_hz": float(f_hz),
                "t_ms": waveform.t.copy(),
                "wave_mT_m": waveform.wave.copy(),
            }
        )

        phase_sig_all_dirs = sim.simulate_multi_directions(waveform)
        phase_sig_radial = phase_sig_all_dirs[0, :]
        phase_sig_axial = phase_sig_all_dirs[1, :]

        # Use full active block duration for the analytical comparison proxy.
        sigma_ms = float(waveform.T_duration)
        b_base = waveform.calculate_bvalue_from_wave()
        gmax = np.sqrt((bval_s_mm2 / 1000.0) / b_base)
        gmax_extended = np.sqrt((bval_s_mm2_extended / 1000.0) / b_base)
        bval_ms_um2 = (gmax**2) * b_base

        sim_signal = np.zeros_like(bval_s_mm2)
        sim_signal_axial = np.zeros_like(bval_s_mm2)
        for idx, g in enumerate(gmax):
            sim_signal[idx] = np.sum(np.cos(g * phase_sig_radial), axis=-1) / num_spins
            sim_signal_axial[idx] = np.sum(np.cos(g * phase_sig_axial), axis=-1) / num_spins

        sim_signal_extended = np.zeros_like(bval_s_mm2_extended)
        for idx, g in enumerate(gmax_extended):
            sim_signal_extended[idx] = np.sum(np.cos(g * phase_sig_radial), axis=-1) / num_spins

        tau_ms = sigma_ms
        ana_signal = _cosine_ogse_signal_eq11(
            gmax=gmax,
            frequency_hz=f_hz,
            sigma_ms=sigma_ms,
            tau_ms=tau_ms,
            radius_um=radius_um,
            diffusivity=diffusivity,
            n_roots=n_roots,
            beta=beta_roots,
        )
        ana_signal_extended = _cosine_ogse_signal_eq11(
            gmax=gmax_extended,
            frequency_hz=f_hz,
            sigma_ms=sigma_ms,
            tau_ms=tau_ms,
            radius_um=radius_um,
            diffusivity=diffusivity,
            n_roots=n_roots,
            beta=beta_roots,
        )
        ana_signal_axial = np.exp(-bval_ms_um2 * diffusivity)

        ax_signal.plot(bval_s_mm2, sim_signal, "o", markersize=6, label=f"MC {f_target_hz:.0f} Hz")
        ax_signal.plot(
            bval_s_mm2,
            ana_signal,
            linewidth=2,
            label=f"Analytical proxy {f_target_hz:.0f} Hz (actual {f_hz:.0f} Hz)",
        )

        ax_axial.plot(bval_s_mm2, sim_signal_axial, "o", markersize=6, label=f"MC {f_target_hz:.0f} Hz")
        ax_axial.plot(
            bval_s_mm2,
            ana_signal_axial,
            linewidth=2,
            label=f"Analytical axial {f_target_hz:.0f} Hz (actual {f_hz:.0f} Hz)",
        )

        ax_radial_extended.plot(
            bval_s_mm2_extended,
            sim_signal_extended,
            "o",
            markersize=6,
            label=f"MC {f_target_hz:.0f} Hz",
        )
        ax_radial_extended.plot(
            bval_s_mm2_extended,
            ana_signal_extended,
            linewidth=2,
            label=f"Analytical proxy {f_target_hz:.0f} Hz (actual {f_hz:.0f} Hz)",
        )

        residual = sim_signal - ana_signal
        residual_axial = sim_signal_axial - ana_signal_axial
        rmse = np.sqrt(np.mean(residual**2))
        rmse_axial = np.sqrt(np.mean(residual_axial**2))

        summary_lines.append(
            f"f target={f_target_hz:.1f} Hz, actual={f_hz:.3f} Hz, "
            f"t_eff={waveform.get_effective_diffusion_time():.6f} ms, N={n_cycles}, "
            f"sigma={sigma_ms:.6f} ms, tau={tau_ms:.6f} ms, "
            f"RMSE_radial={rmse:.6f}, RMSE_axial={rmse_axial:.6f}"
        )

    ax_signal.set_xlabel("b-value (s/mm^2)")
    ax_signal.set_ylabel("DWI signal")
    ax_signal.grid(True, linestyle="--", alpha=0.5)
    ax_signal.set_xlim(0.0, 500.0)
    ax_signal.set_ylim(0.6, 1.0)
    ax_signal.set_title("Test06 radial: Monte Carlo vs analytical proxy (0 to 500)")
    ax_signal.legend(frameon=True, fontsize=9)
    fig_signal.tight_layout()
    fig_signal.savefig(os.path.join(plot_dir, "test_06_apodized_ogse_radial_mc_vs_analytical_0_to_500.png"), dpi=150)
    plt.close(fig_signal)

    ax_axial.set_xlabel("b-value (s/mm^2)")
    ax_axial.set_ylabel("DWI signal")
    ax_axial.grid(True, linestyle="--", alpha=0.5)
    ax_axial.set_xlim(0.0, 500.0)
    ax_axial.set_ylim(0.6, 1.0)
    ax_axial.set_title("Test06 axial: Monte Carlo vs analytical OGSE signal (0 to 500)")
    ax_axial.legend(frameon=True, fontsize=9)
    fig_axial.tight_layout()
    fig_axial.savefig(os.path.join(plot_dir, "test_06_apodized_ogse_axial_mc_vs_analytical_0_to_500.png"), dpi=150)
    plt.close(fig_axial)

    ax_radial_extended.set_xlabel("b-value (s/mm^2)")
    ax_radial_extended.set_ylabel("DWI signal")
    ax_radial_extended.grid(True, linestyle="--", alpha=0.5)
    ax_radial_extended.set_xlim(0.0, 3000.0)
    ax_radial_extended.set_ylim(0.0, 1.0)
    ax_radial_extended.set_title("Test06 radial: Monte Carlo vs analytical proxy (0 to 3000)")
    ax_radial_extended.legend(frameon=True, fontsize=9)
    fig_radial_extended.tight_layout()
    fig_radial_extended.savefig(
        os.path.join(plot_dir, "test_06_apodized_ogse_radial_mc_vs_analytical_0_to_3000.png"),
        dpi=150,
    )
    plt.close(fig_radial_extended)

    _plot_used_ogse_waveforms(waveform_records=waveform_records, output_dir=plot_dir)
    with open(os.path.join(plot_dir, "test_06_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    final_spins = sim.spins_d.get()
    _plot_final_spin_positions_3d_all(
        spins_xyz=final_spins,
        lx=lx,
        ly=ly,
        lz=lz,
        output_dir=plot_dir,
    )

    assert len(summary_lines) == 2 + len(frequency_targets_hz)


if __name__ == "__main__":
    test_radial_diffusion_signal_perpendicular_multi_cylinder_apodized_ogse_temporal_grid_5x5()
