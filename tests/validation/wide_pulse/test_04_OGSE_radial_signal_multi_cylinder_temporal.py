import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
from scipy.special import jnp_zeros
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import pycuda.autoinit

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from simulation_toolkit.simulation_engine.diffsim3d import DwiSim3d
from simulation_toolkit.simulation_engine.geometry import SimGeometry3D, Structure3D
from simulation_toolkit.simulation_engine.waveforms import CosineOGSEWaveform


def _reposition_cosine_waveform_blocks(waveform, first_start_ms, inter_block_gap_ms):
    """Reposition non-apodized OGSE blocks to explicit timing: block1, gap, block2."""
    t_duration_ms = waveform.T_duration
    second_start_ms = first_start_ms + t_duration_ms + inter_block_gap_ms
    second_end_ms = second_start_ms + t_duration_ms
    if second_end_ms > waveform.te + 1e-9:
        raise ValueError(
            "Requested OGSE timing exceeds TE: "
            f"second_end={second_end_ms:.3f} ms, TE={waveform.te:.3f} ms"
        )

    info = waveform.get_waveform_info()
    cosine_duration = info["cosine_duration_ms"]
    f_cosine = info["cosine_frequency_Hz"] / 1000.0

    waveform.wave[:] = 0.0
    waveform._generate_single_waveform(first_start_ms, 1.0, cosine_duration, f_cosine)
    waveform._generate_single_waveform(second_start_ms, -1.0, cosine_duration, f_cosine)


def _plot_used_ogse_waveforms(waveform_records, output_dir):
    """Plot OGSE waveforms used in test04 as separate figures."""
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
            f"Test04 waveform: target {rec['f_target_hz']:.0f} Hz (actual {rec['f_actual_hz']:.0f} Hz)"
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f"test_04_ogse_waveform_target_{rec['f_target_hz']:.0f}Hz_actual_{rec['f_actual_hz']:.0f}Hz.png",
            ),
            dpi=150,
        )
        plt.close(fig)


def _build_single_periodic_cylinder_geometry(
    lx,
    ly,
    lz,
    center_xy,
    radius,
    diffusivity,
    t2,
    rho,
    n_segments=200,
    extension_segments=30,
):
    """Build a single z-aligned impermeable cylinder with periodic z extension."""
    sg3 = SimGeometry3D(lx, ly, lz, diffusivity, t2, rho)

    base_z = np.linspace(-lz / 2.0, lz / 2.0, num=n_segments)
    cx, cy = center_xy

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


def _cylinder_perp_diffusion_spectrum_hz(frequency_hz, radius_um, diffusivity, n_roots=120, beta=None):
    """Approximate D_perp(omega) for an impermeable cylinder from modal poles.

    This is a finite-mode diffusion-spectrum approximation where each mode has
    relaxation rate lambda_k = D * (beta_k / R)^2 and beta_k are roots of J1'.
    Coefficients are normalized so D(0)=0 and D(omega->inf)=D.
    """
    if beta is None:
        beta = jnp_zeros(1, n_roots)
    weights = 1.0 / (beta**2 * (beta**2 - 1.0))
    weights = weights / np.sum(weights)

    omega_rad_ms = 2.0 * np.pi * float(frequency_hz) / 1000.0
    lambdas = diffusivity * (beta / radius_um) ** 2  # 1/ms
    modal_terms = (omega_rad_ms**2) / (omega_rad_ms**2 + lambdas**2)
    return diffusivity * np.sum(weights * modal_terms)


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
    """Analytical OGSE signal using Eq. (10)+(11) style attenuation.

    Implements E(2*tau)=exp(-beta(2*tau)) with the modal sum over cylinder
    eigenmodes for cosine OGSE.
    """
    if beta is None:
        beta = jnp_zeros(1, n_roots)

    g_arr = np.asarray(gmax, dtype=float)
    gamma = 267.513  # rad/(ms*mT)
    # Simulator gradient scaling is in mT/m, while Eq. (11) terms here use um-based
    # geometry/diffusivity units. Convert to mT/um for a consistent gamma*g unit.
    g_mT_um = g_arr * 1e-6  # mT/m -> mT/um

    omega = 2.0 * np.pi * float(frequency_hz) / 1000.0  # rad/ms
    mu_n = np.asarray(beta, dtype=float)
    # Eq. (6): B_n = 2*(R/mu_n)^2/(mu_n^2 - 1), lambda_n = (mu_n/R)^2
    b_n = 2.0 * (radius_um / mu_n) ** 2 / (mu_n**2 - 1.0)
    lambda_n = (mu_n / radius_um) ** 2  # 1/um^2

    lam_d = lambda_n * diffusivity
    lam2_d2 = (lambda_n**2) * (diffusivity**2)
    
    if abs(omega) < 1e-14:
        trig_term = sigma_ms
    else:
        trig_term = sigma_ms / 2.0 + np.sin(2.0 * omega * sigma_ms) / (4.0 * omega)

    # Numerically stable rewrite of exp(-a) * (1 - cosh(b)):
    # exp(-a) - 0.5*exp(-(a-b)) - 0.5*exp(-(a+b))
    # This avoids overflow from cosh for large modes.
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
    signal = np.exp(-beta_2tau)
    return signal


def _cylinder_perp_diffusion_spectrum_from_omega_rad_ms(omega_rad_ms, radius_um, diffusivity, n_roots=120, beta=None):
    """Vectorized D_perp(omega) for an impermeable cylinder over angular frequencies."""
    if beta is None:
        beta = jnp_zeros(1, n_roots)
    weights = 1.0 / (beta**2 * (beta**2 - 1.0))
    weights = weights / np.sum(weights)

    omega = np.asarray(omega_rad_ms, dtype=float)
    lambdas = diffusivity * (beta / radius_um) ** 2  # 1/ms
    modal_terms = (omega[:, None] ** 2) / (omega[:, None] ** 2 + lambdas[None, :] ** 2)
    return diffusivity * np.sum(weights[None, :] * modal_terms, axis=1)


def _effective_dapp_from_waveform_spectrum(waveform, radius_um, diffusivity, n_roots=120, beta=None):
    """Compute waveform-weighted effective Dapp via the temporal spectrum.

    This is a GPA-style frequency-domain reduction:
    Dapp = sum(|Q(omega)|^2 * D(omega)) / sum(|Q(omega)|^2),
    where Q(omega) is the Fourier transform of q(t)=int g(t)dt.
    """
    q_t = np.cumsum(waveform.wave) * waveform.dt
    q_t = q_t - np.mean(q_t)

    q_fft = np.fft.rfft(q_t)
    freqs_hz = np.fft.rfftfreq(q_t.size, d=waveform.dt / 1000.0)

    # Ignore DC component for weighting.
    weights = np.abs(q_fft) ** 2
    weights[0] = 0.0

    valid = weights > 0
    if not np.any(valid):
        raise RuntimeError("Waveform spectrum has no non-zero frequency content for Dapp weighting.")

    omega_rad_ms = 2.0 * np.pi * freqs_hz[valid] / 1000.0
    d_omega = _cylinder_perp_diffusion_spectrum_from_omega_rad_ms(
        omega_rad_ms=omega_rad_ms,
        radius_um=radius_um,
        diffusivity=diffusivity,
        n_roots=n_roots,
        beta=beta,
    )
    w = weights[valid]
    dapp_eff = float(np.sum(w * d_omega) / np.sum(w))
    return dapp_eff


def _normalize_spin_array(spins_xyz):
    """Return spin positions as an array shaped (nspins, 3)."""
    spins_xyz = np.asarray(spins_xyz, dtype=float)
    if spins_xyz.ndim != 2:
        raise ValueError("spins_xyz must be a 2D array.")

    # Common cases:
    # - (nspins, 3) -> keep as-is
    # - (3, nspins) -> transpose
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
    ax.set_title("Test04: Final spin positions (3D, all spins) after OGSE simulation")
    ax.view_init(elev=24, azim=42)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_04_final_spin_positions_3d_all.png"), dpi=170)
    plt.close(fig)


def test_radial_diffusion_signal_perpendicular_multi_cylinder_cosine_ogse_temporal():
    """Compare Monte Carlo and analytical OGSE signals for a single cylinder."""
    diffusivity = 1.0  # um^2/ms (intra-axonal)
    cylinder_radius_um = 0.98

    # OGSE design: TE = gradient on-time + gradient on-time.
    frequency_targets_hz = [100.0, 250.0, 500.0, 1000.0]
    t_duration_ms = 20.0
    te_ms = 2.0 * t_duration_ms
    first_block_start_ms = 0.0
    inter_block_gap_ms = 0.0
    if t_duration_ms <= 0:
        raise ValueError("Invalid OGSE timing: T_duration must be positive.")
    if (first_block_start_ms + 2.0 * t_duration_ms + inter_block_gap_ms) > te_ms + 1e-9:
        raise ValueError("Invalid OGSE timing: start + 2*T_duration + gap must be <= TE.")

    output_folder = "./tests/validation/wide_pulse/test_figures"
    folder_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_dir = os.path.join(output_folder, folder_date_time)
    os.makedirs(plot_dir, exist_ok=True)

    sg3 = _build_single_periodic_cylinder_geometry(
        lx=20.0,
        ly=20.0,
        lz=20.0,
        center_xy=(0.0, 0.0),
        radius=cylinder_radius_um,
        diffusivity=diffusivity,
        t2=200,
        rho=1.0,
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
    sim.set_segments(nsegx=5, nsegy=5, nsegz=5)
    sim.setup(structures=list(np.arange(0, sg3.nstructures)))

    # Precompute Bessel roots once for all OGSE cases.
    n_roots = 200
    beta_roots = jnp_zeros(1, n_roots)

    # Base b-value grid: 11 points from 0 to 500 s/mm^2.
    bval_s_mm2 = np.linspace(0.0, 500.0, 11)
    # Extended radial figure grid: 0-500 (11 points) plus 1000..3000 in 500 steps.
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
    summary_lines = ["Test04 OGSE Monte Carlo vs analytical signal summary", "=" * 52]

    for f_target_hz in frequency_targets_hz:
        sim.reset_simulation()
        n_cycles = int(round(f_target_hz * t_duration_ms / 1000.0))
        if n_cycles <= 0:
            raise ValueError("Computed n_cycles must be positive.")

        # Keep T fixed while varying N to modulate frequency.
        waveform = CosineOGSEWaveform(
            N_cycles=n_cycles,
            T_duration=t_duration_ms,
            te=te_ms,
            gmax=1.0,
            time_step=0.01,
        )

        # Enforce explicit timing requested for visualization: first block,
        # optional gap, second block.
        _reposition_cosine_waveform_blocks(
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

        sigma_ms = float(info["cosine_duration_ms"])
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

        tau_ms = sigma_ms # For this cosine OGSE, tau = T-duration = sigma_ms
        ana_signal = _cosine_ogse_signal_eq11(
            gmax=gmax,
            frequency_hz=f_hz,
            sigma_ms=sigma_ms,
            tau_ms=tau_ms,
            radius_um=cylinder_radius_um,
            diffusivity=diffusivity,
            n_roots=n_roots,
        )
        ana_signal_extended = _cosine_ogse_signal_eq11(
            gmax=gmax_extended,
            frequency_hz=f_hz,
            sigma_ms=sigma_ms,
            tau_ms=tau_ms,
            radius_um=cylinder_radius_um,
            diffusivity=diffusivity,
            n_roots=n_roots,
        )
        if not np.all(np.isfinite(ana_signal)):
            raise ValueError(
                "Analytical OGSE signal has NaN/Inf values; check Eq.11 parameterization and units."
            )
        if not np.all(np.isfinite(ana_signal_extended)):
            raise ValueError(
                "Extended analytical OGSE signal has NaN/Inf values; check Eq.11 parameterization and units."
            )
        ana_signal_axial = np.exp(-bval_ms_um2 * diffusivity)

        ax_signal.plot(
            bval_s_mm2,
            sim_signal,
            "o",
            markersize=6,
            label=f"MC {f_target_hz:.0f} Hz",
        )
        ax_signal.plot(
            bval_s_mm2,
            ana_signal,
            linewidth=2,
            label=f"Analytical {f_target_hz:.0f} Hz (actual {f_hz:.0f} Hz)",
        )

        ax_axial.plot(
            bval_s_mm2,
            sim_signal_axial,
            "o",
            markersize=6,
            label=f"MC {f_target_hz:.0f} Hz",
        )
        ax_axial.plot(
            bval_s_mm2,
            ana_signal_axial,
            linewidth=2,
            label=f"Analytical {f_target_hz:.0f} Hz (actual {f_hz:.0f} Hz)",
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
            label=f"Analytical {f_target_hz:.0f} Hz (actual {f_hz:.0f} Hz)",
        )

        residual = sim_signal - ana_signal
        rmse = np.sqrt(np.mean(residual ** 2))
        residual_axial = sim_signal_axial - ana_signal_axial
        rmse_axial = np.sqrt(np.mean(residual_axial ** 2))

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
    ax_signal.set_title("Test04: Monte Carlo vs analytical OGSE signal across frequencies")
    ax_signal.legend(frameon=True, fontsize=9)
    fig_signal.tight_layout()
    fig_signal.savefig(os.path.join(plot_dir, "test_04_ogse_mc_vs_analytical_0_to_500.png"), dpi=150)
    plt.close(fig_signal)

    ax_axial.set_xlabel("b-value (s/mm^2)")
    ax_axial.set_ylabel("DWI signal")
    ax_axial.grid(True, linestyle="--", alpha=0.5)
    ax_axial.set_xlim(0.0, 500.0)
    ax_axial.set_ylim(0.6, 1.0)
    ax_axial.set_title("Test04 axial: Monte Carlo vs analytical OGSE signal across frequencies")
    ax_axial.legend(frameon=True, fontsize=9)
    fig_axial.tight_layout()
    fig_axial.savefig(os.path.join(plot_dir, "test_04_ogse_axial_mc_vs_analytical_0_to_500.png"), dpi=150)
    plt.close(fig_axial)

    ax_radial_extended.set_xlabel("b-value (s/mm^2)")
    ax_radial_extended.set_ylabel("DWI signal")
    ax_radial_extended.grid(True, linestyle="--", alpha=0.5)
    ax_radial_extended.set_xlim(0.0, 3000.0)
    ax_radial_extended.set_ylim(0.0, 1.0)
    ax_radial_extended.set_title("Test04 radial: Monte Carlo vs analytical OGSE signal (0 to 3000)")
    ax_radial_extended.legend(frameon=True, fontsize=9)
    fig_radial_extended.tight_layout()
    fig_radial_extended.savefig(
        os.path.join(plot_dir, "test_04_ogse_radial_mc_vs_analytical_0_to_3000.png"),
        dpi=150,
    )
    plt.close(fig_radial_extended)

    _plot_used_ogse_waveforms(waveform_records=waveform_records, output_dir=plot_dir)
    with open(os.path.join(plot_dir, "test_04_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    final_spins = sim.spins_d.get()
    _plot_final_spin_positions_3d_all(
        spins_xyz=final_spins,
        lx=20.0,
        ly=20.0,
        lz=20.0,
        output_dir=plot_dir,
    )

    assert len(summary_lines) == 2 + len(frequency_targets_hz)


if __name__ == "__main__":
    test_radial_diffusion_signal_perpendicular_multi_cylinder_cosine_ogse_temporal()
