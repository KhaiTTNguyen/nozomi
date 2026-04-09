#!/usr/bin/env python3
"""Plot non-apodized cosine OGSE waveforms for selected frequencies."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add the simulation toolkit to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from simulation_toolkit.simulation_engine.waveforms import CosineOGSEWaveform


def _reposition_cosine_waveform_blocks(waveform, first_start_ms, inter_block_gap_ms):
    """Reposition two cosine OGSE blocks to explicit timing."""
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


def plot_cosine_ogse_examples():
    """Plot cosine OGSE waveforms for 100/250/500/1000 Hz."""
    frequency_targets_hz = [100.0, 250.0, 500.0, 1000.0]

    # Use TE = T + T (no margins/gap), so T=20 ms and TE=40 ms.
    t_duration_ms = 20.0
    te_ms = 2.0 * t_duration_ms
    first_block_start_ms = 0.0
    inter_block_gap_ms = 0.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, f_target_hz in enumerate(frequency_targets_hz):
        n_cycles = int(round(f_target_hz * t_duration_ms / 1000.0))
        waveform = CosineOGSEWaveform(
            N_cycles=n_cycles,
            T_duration=t_duration_ms,
            te=te_ms,
            gmax=1.0,
            time_step=0.001,
        )
        _reposition_cosine_waveform_blocks(
            waveform=waveform,
            first_start_ms=first_block_start_ms,
            inter_block_gap_ms=inter_block_gap_ms,
        )

        info = waveform.get_waveform_info()
        f_actual_hz = info["cosine_frequency_Hz"]

        ax = axes[idx]
        ax.plot(waveform.t, waveform.wave, linewidth=1.8)
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Normalized gradient")
        ax.set_title(
            f"f target={f_target_hz:.0f} Hz, actual={f_actual_hz:.1f} Hz\n"
            f"N={n_cycles}, T={t_duration_ms:.1f} ms, TE={te_ms:.1f} ms"
        )

    plt.tight_layout()
    plt.suptitle("Non-apodized cosine OGSE waveforms", y=0.995, fontsize=14)
    out_png = "./tests/validation/wide_pulse/development/cosine_ogse_examples.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    plot_cosine_ogse_examples()


if __name__ == "__main__":
    main()
