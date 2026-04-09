#!/usr/bin/env python3
"""
Plot example trapezoidal-cosine OGSE waveforms.

Examples follow requested timing:
- Gradient on-time per block T = 40.9 ms
- Separation between blocks = 51.4 ms
- N = 1 and N = 2 cosine oscillations

Effective diffusion time convention:
    t_eff = T / (4N)
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add the simulation toolkit to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from simulation_toolkit.simulation_engine.waveforms import TrapezoidalCosineOGSEWaveform


def _build_timing_segments(block_start, n_cycles, trise, tp, t3):
    """Return (start, end, kind) timing segments for one trapezoid-cosine block."""
    segments = []
    cursor = block_start

    # Initial ramp: tr
    segments.append((cursor, cursor + trise, "tr"))
    cursor += trise

    n_plateaus = 2 * n_cycles + 1
    for idx in range(n_plateaus):
        if idx == 0 or idx == (n_plateaus - 1):
            segments.append((cursor, cursor + tp, "tp"))
            cursor += tp
        else:
            # Interior OGSE plateau is 2*t3; split into two t3 pieces for QC display.
            segments.append((cursor, cursor + t3, "t3"))
            cursor += t3
            segments.append((cursor, cursor + t3, "t3"))
            cursor += t3

        if idx < (n_plateaus - 1):
            # Transition is 2*tr; show as two tr segments.
            segments.append((cursor, cursor + trise, "tr"))
            cursor += trise
            segments.append((cursor, cursor + trise, "tr"))
            cursor += trise

    # Final ramp: tr
    segments.append((cursor, cursor + trise, "tr"))
    return segments


def plot_trapezoid_ogse_examples():
    T_duration = 40.9
    separation = 51.4
    N_values = [1, 2]

    # Xu-style timing uses tr, tp, and t3 = tp + tr/2.
    # Use a single tr for both N=1 and N=2 examples and let tp be derived from T.
    trise = 0.9
    full_gradient_duration = separation + T_duration
    te = full_gradient_duration

    print("Trapezoidal cosine OGSE examples")
    print("=" * 40)
    print(f"T_duration: {T_duration:.1f} ms")
    print(f"Separation: {separation:.1f} ms")
    print(f"Full gradient duration: {full_gradient_duration:.1f} ms")
    print(f"TE: {te:.1f} ms")
    print(f"trise: {trise:.1f} ms")
    print()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    duration_colors = {"tr": "tab:orange", "tp": "tab:green", "t3": "tab:purple"}

    for ax, N in zip(axes, N_values):
        waveform = TrapezoidalCosineOGSEWaveform(
            N_cycles=N,
            T_duration=T_duration,
            separation_duration=separation,
            trise=trise,
            tp=None,
            te=te,
            gmax=1.0,
            time_step=0.001,
        )

        info = waveform.get_waveform_info()

        ax.plot(waveform.t, waveform.wave, linewidth=1.4, color="0.5", alpha=0.7)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

        b1s = info["block1_start_ms"]
        b2s = info["block2_start_ms"]

        for block_start in (b1s, b2s):
            for t0, t1, kind in _build_timing_segments(
                block_start=block_start,
                n_cycles=N,
                trise=info["trise_ms"],
                tp=info["tp_ms"],
                t3=info["t3_ms"],
            ):
                mask = (waveform.t >= t0) & (waveform.t < t1)
                ax.plot(
                    waveform.t[mask],
                    waveform.wave[mask],
                    linewidth=2.8,
                    color=duration_colors[kind],
                    solid_capstyle="round",
                )

        ax.set_ylabel("Normalized G")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(
            f"Trapezoidal Cosine OGSE: N={N}, t_eff={waveform.get_effective_diffusion_time():.4f} ms"
        )

        legend_handles = [
            Line2D([0], [0], color=duration_colors["tr"], lw=3, label="tr"),
            Line2D([0], [0], color=duration_colors["tp"], lw=3, label="tp"),
            Line2D([0], [0], color=duration_colors["t3"], lw=3, label="t3"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)

        print(
            f"N={N}: t_eff={waveform.get_effective_diffusion_time():.6f} ms, "
            f"tr={info['trise_ms']:.4f} ms, tp={info['tp_ms']:.4f} ms, t3={info['t3_ms']:.4f} ms, "
            f"OGSE plateau={info['ogse_plateau_ms']:.4f} ms, "
            f"f={info['cosine_frequency_Hz']:.3f} Hz"
        )

    axes[-1].set_xlabel("Time (ms)")
    plt.suptitle("Trapezoidal-cosine OGSE waveform examples", y=0.98)
    plt.tight_layout()

    output_path = "./tests/validation/wide_pulse/development/trapezoid_ogse_examples.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print()
    print(f"Saved plot to: {output_path}")


def plot_trapezoid_ogse_teff_series():
    """Plot trapezoidal-cosine OGSE waveforms for N=2/4/6 at T=40 ms."""
    T_duration = 40.0
    separation = 50.0
    # Use a single tr for all N, constrained by N=8 validity under
    # tp = 1/4 * (T/N - 13*tr/2). This requires tr < (T/N)/6.5 = 0.769... ms.
    trise = 0.6
    N_values = [2, 4, 6]

    if trise <= 0.1:
        raise ValueError("Second figure requires trise > 0.1 ms.")

    full_gradient_duration = separation + T_duration
    te = full_gradient_duration

    print("\nTrapezoidal cosine OGSE t_eff series")
    print("=" * 40)
    print(f"T_duration: {T_duration:.1f} ms")
    print(f"Separation: {separation:.1f} ms")
    print(f"TE: {te:.1f} ms")
    print(f"trise: {trise:.1f} ms")

    fig, axes = plt.subplots(len(N_values), 1, figsize=(14, 10), sharex=True)
    duration_colors = {"tr": "tab:orange", "tp": "tab:green", "t3": "tab:purple"}

    for ax, N in zip(axes, N_values):
        waveform = TrapezoidalCosineOGSEWaveform(
            N_cycles=N,
            T_duration=T_duration,
            separation_duration=separation,
            trise=trise,
            tp=None,
            te=te,
            gmax=1.0,
            time_step=0.001,
        )
        info = waveform.get_waveform_info()

        ax.plot(waveform.t, waveform.wave, linewidth=1.4, color="0.5", alpha=0.7)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

        for block_start in (info["block1_start_ms"], info["block2_start_ms"]):
            for t0, t1, kind in _build_timing_segments(
                block_start=block_start,
                n_cycles=N,
                trise=info["trise_ms"],
                tp=info["tp_ms"],
                t3=info["t3_ms"],
            ):
                mask = (waveform.t >= t0) & (waveform.t < t1)
                ax.plot(
                    waveform.t[mask],
                    waveform.wave[mask],
                    linewidth=2.6,
                    color=duration_colors[kind],
                    solid_capstyle="round",
                )

        ax.set_ylabel("Normalized G")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(
            f"Trapezoidal Cosine OGSE: N={N}, t_eff={waveform.get_effective_diffusion_time():.4f} ms"
        )

        legend_handles = [
            Line2D([0], [0], color=duration_colors["tr"], lw=3, label="tr"),
            Line2D([0], [0], color=duration_colors["tp"], lw=3, label="tp"),
            Line2D([0], [0], color=duration_colors["t3"], lw=3, label="t3"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)

        print(
            f"N={N}: t_eff={waveform.get_effective_diffusion_time():.4f} ms "
            f"(target {T_duration/(4.0*N):.4f} ms), "
            f"tr={info['trise_ms']:.4f} ms, tp={info['tp_ms']:.4f} ms, t3={info['t3_ms']:.4f} ms"
        )

    axes[-1].set_xlabel("Time (ms)")
    plt.suptitle("Trapezoidal-cosine OGSE waveforms (N=2/4/6, T=40 ms)", y=0.98)
    plt.tight_layout()

    output_path = "./tests/validation/wide_pulse/development/trapezoid_ogse_teff_series.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    plot_trapezoid_ogse_examples()
    plot_trapezoid_ogse_teff_series()
