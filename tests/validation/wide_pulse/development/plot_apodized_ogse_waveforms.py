#!/usr/bin/env python3
"""
Plot apodized cosine modulated oscillating gradient waveforms
Based on Does et al. 2003 implementation

This script generates examples with N cycles [1,3,5,8,10,13] to achieve
t_eff ranging from 6.5 ms down to 0.5 ms as described in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the simulation toolkit to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
from simulation_toolkit.simulation_engine.waveforms import ApodizedCosineOGSEWaveform


def _reposition_apodized_waveform_blocks(waveform, first_start_ms, inter_block_gap_ms):
    """Reposition OGSE blocks to explicit timing: block1, gap, block2."""
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

def plot_apodized_ogse_examples():
    """
    Plot examples of apodized cosine OGSE waveforms with different N cycles
    """
    # Parameters from Does et al. paper
    N_values = [1, 3, 5, 8, 10, 13]
    target_t_eff_range = [6.5, 0.5]  # ms
    t_ontime = 26.0
    inter_block_gap = 5.0
    te_total = 2.0 * t_ontime + inter_block_gap
    
    print("Apodized Cosine OGSE waveforms")
    print("=" * 50)
    print(f"Target t_eff range: {target_t_eff_range[0]} ms to {target_t_eff_range[1]} ms")
    print()
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, N in enumerate(N_values):        
        T_ontime = t_ontime
        
        print(f"N = {N} cycles:")
        print(f"  Gradient on-time: T = {T_ontime:.1f} ms, t_eff = {T_ontime/(4*N):.2f} ms")
        print()
        
        # Generate waveform
        waveform = ApodizedCosineOGSEWaveform(
            N_cycles=N, 
            T_duration=T_ontime,
            te=te_total,
            gmax=1.0,
            time_step=0.001  # 1 μs resolution
        )
        _reposition_apodized_waveform_blocks(
            waveform=waveform,
            first_start_ms=0.0,
            inter_block_gap_ms=inter_block_gap,
        )
        
        # Plot
        ax = axes[i]
        ax.plot(waveform.t, waveform.wave, 'b-', linewidth=1.5)
        ax.set_title(
            f'N = {N} cycles, t_eff = {waveform.get_effective_diffusion_time():.2f} ms\n'
            f'T = {T_ontime:.1f} ms, gap = {inter_block_gap:.1f} ms, TE = {te_total:.1f} ms'
        )
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Normalized Gradient Amplitude')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.2, 1.2)
        
        # Highlight the active gradient regions
        info = waveform.get_waveform_info()
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.suptitle('Apodized Cosine OGSE waveforms (TE = 26 + 5 + 26 ms)', y=0.98, fontsize=14)
    plt.savefig('./tests/validation/wide_pulse/development/apodized_ogse_examples.png', dpi=500, bbox_inches='tight')
    plt.show()

def plot_single_waveform_detailed(N=5, t_eff_target=2.0):
    """
    Plot a detailed view of a single apodized OGSE waveform
    """
    T = 4 * N * t_eff_target
    
    inter_block_gap = 5.0
    te_total = 2.0 * T + inter_block_gap

    waveform = ApodizedCosineOGSEWaveform(
        N_cycles=N, 
        T_duration=T,
        te=te_total,
        gmax=1.0,
        time_step=0.0001  # High resolution for smooth curves
    )
    _reposition_apodized_waveform_blocks(
        waveform=waveform,
        first_start_ms=0.0,
        inter_block_gap_ms=inter_block_gap,
    )
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Full waveform
    ax1.plot(waveform.t, waveform.wave, 'b-', linewidth=2, label='Full Waveform')
    ax1.set_title(f'Apodized Cosine OGSE Waveform (N={N}, t_eff={t_eff_target} ms)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Normalized Gradient Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Zoomed view of encoding portion
    info = waveform.get_waveform_info()
    t_start_encode = waveform.te/4 - T/2
    t_end_encode = t_start_encode + T
    
    # Find indices for zoom
    zoom_mask = (waveform.t >= t_start_encode - 0.5) & (waveform.t <= t_end_encode + 0.5)
    
    ax2.plot(waveform.t[zoom_mask], waveform.wave[zoom_mask], 'b-', linewidth=2)
    ax2.set_title('Encoding Portion (Detailed View)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Normalized Gradient Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for different phases
    ax2.axvline(x=t_start_encode, color='r', linestyle='--', alpha=0.7, label='Start')
    ax2.axvline(x=t_end_encode, color='r', linestyle='--', alpha=0.7, label='End')
    
    # Calculate phase boundaries for half-sine(2f) endcaps and
    # negative-start/negative-end cosine middle segment.
    f_cos = N / T
    endcap_dur = 0.25 / f_cos
    cosine_dur = (N - 0.5) / f_cos
    
    phase1_end = t_start_encode + endcap_dur
    phase2_end = phase1_end + cosine_dur
    
    ax2.axvline(x=phase1_end, color='g', linestyle=':', alpha=0.7, label='half-sine(2f) → cosines')
    ax2.axvline(x=phase2_end, color='g', linestyle=':', alpha=0.7, label='cosines → half-sine(2f)')
    
    ax2.legend()
    
    # Print waveform information
    print("\nDetailed Waveform Information:")
    print("=" * 30)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    plt.tight_layout()
    plt.savefig('/tests/validation/wide_pulse/development/apodized_ogse_detailed.png', dpi=500, bbox_inches='tight')
    plt.show()

def analyze_waveform_parameters():
    """
    Analyze the relationship between N, T, and t_eff for the paper's examples
    """
    N_values = [1, 3, 5, 8, 10, 13]
    t_eff_values = [6.5, 2.17, 1.3, 0.81, 0.65, 0.5]  # Example values
    
    print("\nWaveform Parameter Analysis")
    print("=" * 40)
    print("N\tt_eff (ms)\tT (ms)\tf_cos (Hz)\tf_sine (Hz)")
    print("-" * 50)
    
    for N, t_eff in zip(N_values, t_eff_values):
        T = 4 * N * t_eff
        f_cos = N / T * 1000  # Convert to Hz
        f_sine = 2 * f_cos
        
        print(f"{N}\t{t_eff:.3f}\t\t{T:.1f}\t\t{f_cos:.1f}\t\t{f_sine:.1f}")
    
    # Plot parameter relationships
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # T vs N for different t_eff values
    t_eff_range = [6.5, 2.17, 1.3, 0.81, 0.65, 0.5]
    N_range = np.array(N_values)
    
    for t_eff in t_eff_range:
        T_values = 4 * N_range * t_eff
        ax1.plot(N_range, T_values, 'o-', label=f't_eff = {t_eff} ms')
    
    ax1.set_xlabel('Number of Cycles (N)')
    ax1.set_ylabel('Active Duration T (ms)')
    ax1.set_title('T vs N for Different t_eff')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Frequency vs N
    T_ontime = 4 * N_range * 2.0  # t_eff = 2.0 ms
    f_cos_values = N_range / T_ontime * 1000
    f_sine_values = 2 * f_cos_values
    
    ax2.plot(N_range, f_cos_values, 'o-', label='Cosine frequency')
    ax2.plot(N_range, f_sine_values, 's-', label='Sine frequency (2x)')
    ax2.set_xlabel('Number of Cycles (N)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Frequencies vs N (t_eff = 2.0 ms)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./tests/validation/wide_pulse/development/apodized_ogse_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all examples"""
    # Plot examples with different N values
    plot_apodized_ogse_examples()
    
    # Detailed view of a single waveform
    # plot_single_waveform_detailed(N=5, t_eff_target=2.0)
    
    # Parameter analysis
    # analyze_waveform_parameters()
    
    print("\nAll plots saved to nozomi/tests/validation/wide_pulse/development/")

if __name__ == "__main__":
    main()