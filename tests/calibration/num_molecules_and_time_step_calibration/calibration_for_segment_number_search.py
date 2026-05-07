"""Quick calibration: run time vs number of segments (nsegx=nsegy=nsegz=N).

Reuses the Gamma-distributed cylinder substrate builder from
``calibration_for_num_molecules_and_time_step_search.py`` and sweeps the
spatial segmentation parameter of ``DiffSim3d.set_segments``. Produces a plot
of tabulation / simulation / total computation time vs number of segments.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import pycuda.gpuarray as gpuarray

import simulation_toolkit.simulation_engine.diffsim3d as ds3
import simulation_toolkit.toolkit_params as config_params

from calibration_for_num_molecules_and_time_step_search import (
    axon_gamma_dist_gen,
    build_edge_ghost_cylinders,
    build_multi_cylinder_geometry,
)


def run_single_segment_timing(sg3, molecules, time_step, total_sim_time, nseg):
    """Return (tabulation_time_s, simulation_time_s) for a given segment count."""
    nt = int(total_sim_time / time_step)
    numsteps = nt + 1

    sim = ds3.DiffSim3d(sg3, int(molecules))

    t0 = time.time()
    sim.set_segments(nsegx=nseg, nsegy=nseg, nsegz=nseg)
    sim.setup(structures=list(np.arange(0, sg3.nstructures)))
    tab_time = time.time() - t0

    Dxarray = gpuarray.zeros(numsteps, dtype=np.float32)
    Dyarray = gpuarray.zeros(numsteps, dtype=np.float32)
    Dzarray = gpuarray.zeros(numsteps, dtype=np.float32)
    Kx2 = gpuarray.zeros(numsteps, dtype=np.float32)
    Ky2 = gpuarray.zeros(numsteps, dtype=np.float32)
    Kz2 = gpuarray.zeros(numsteps, dtype=np.float32)
    Kx4 = gpuarray.zeros(numsteps, dtype=np.float32)
    Ky4 = gpuarray.zeros(numsteps, dtype=np.float32)
    Kz4 = gpuarray.zeros(numsteps, dtype=np.float32)

    current_time = 0.0
    step_idx = 1
    t0 = time.time()
    while current_time < nt * time_step:
        sim.step(time_step)
        current_time += time_step
        sim.calculate_diffusion_coefficients_and_kurtoses(
            step_idx, current_time,
            Dxarray, Dyarray, Dzarray,
            Kx2, Ky2, Kz2, Kx4, Ky4, Kz4,
        )
        step_idx += 1
    sim_time = time.time() - t0

    return tab_time, sim_time


def plot_runtime_vs_segments(segments, tab_times, sim_times, output_path,
                             molecules, time_step, total_sim_time):
    """Stacked area-style plot matching the reference figure."""
    segments = np.asarray(segments)
    tab_times = np.asarray(tab_times)
    sim_times = np.asarray(sim_times)
    total_times = tab_times + sim_times

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.fill_between(segments, 0, tab_times,
                    color="tab:green", alpha=0.35, label="Tabulation Time")
    ax.fill_between(segments, tab_times, total_times,
                    color="tab:blue", alpha=0.25, label="Simulation Time")
    ax.plot(segments, tab_times, "o-", color="tab:green", linewidth=2)
    ax.plot(segments, total_times, "o-", color="blue", linewidth=2.5,
            label="Total Computation Time")

    ax.set_xlabel("Number of Segments", fontsize=16)
    ax.set_ylabel("Run Time (sec)", fontsize=16)
    ax.set_title("Run Time vs Number of Segments", fontsize=17, fontweight="bold")
    ax.grid(True, alpha=0.4)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.legend(fontsize=13, loc="upper right")

    ax.text(
        0.02, 0.98,
        f"molecules = {int(molecules):,}\n"
        f"time step = {time_step} ms\n"
        f"total sim time = {total_sim_time} ms",
        transform=ax.transAxes, fontsize=11, va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def run_segment_calibration():
    # Physical / simulation parameters (lightweight for fast sweep)
    D0 = 2.0
    molecules = int(5e4)
    time_step = 0.002
    total_sim_time = 100

    # Substrate: same Gamma distribution as the main calibration
    kappa = 4.0
    theta_um = 0.45
    n_axons = 500
    axon_area_fraction = 0.65

    # Segment sweep
    segment_values = [10, 20, 30, 40, 50, 60]
    n_repeats = 1

    output_dir = (
        "./tests/calibration/num_molecules_and_time_step_calibration/figs/"
        "segment_calibration_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Build substrate once
    print("Generating Gamma-distributed cylinder substrate...")
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
        lx=lx, ly=ly,
        edge_distance=lxy / 5.0,
    )
    print(f"  Primary: {len(primary_radii)}  Total: {len(radii)}  Box: {lxy:.2f} μm")

    print("Building SimGeometry3D...")
    sg3 = build_multi_cylinder_geometry(centers_xy, radii, lx, ly, lz, D0)
    print(f"  nstructures: {sg3.nstructures}")

    # Sweep
    tab_means, sim_means = [], []
    tab_stds, sim_stds = [], []

    print("\nSweeping segment counts...")
    for nseg in segment_values:
        print(f"\n  nseg = {nseg}")
        tabs, sims = [], []
        for r in range(n_repeats):
            print(f"    repeat {r + 1}/{n_repeats} ...", end=" ", flush=True)
            try:
                tab_t, sim_t = run_single_segment_timing(
                    sg3, molecules, time_step, total_sim_time, nseg
                )
                print(f"tab={tab_t:.2f}s  sim={sim_t:.2f}s")
                tabs.append(tab_t)
                sims.append(sim_t)
            except Exception as e:
                print(f"ERROR: {e}")
        if tabs:
            tab_means.append(np.mean(tabs))
            sim_means.append(np.mean(sims))
            tab_stds.append(np.std(tabs))
            sim_stds.append(np.std(sims))
        else:
            tab_means.append(np.nan)
            sim_means.append(np.nan)
            tab_stds.append(np.nan)
            sim_stds.append(np.nan)

    # Save CSV
    import pandas as pd
    df = pd.DataFrame({
        "n_segments": segment_values,
        "tabulation_time_s": tab_means,
        "simulation_time_s": sim_means,
        "total_time_s": np.array(tab_means) + np.array(sim_means),
        "tabulation_time_std_s": tab_stds,
        "simulation_time_std_s": sim_stds,
    })
    csv_path = os.path.join(output_dir, "segment_calibration.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")

    # Plot
    plot_path = os.path.join(output_dir, "segment_calibration_plot.png")
    plot_runtime_vs_segments(
        segment_values, tab_means, sim_means, plot_path,
        molecules, time_step, total_sim_time,
    )

    # Print recommendation
    totals = np.array(tab_means) + np.array(sim_means)
    best_idx = int(np.nanargmin(totals))
    print(f"\nOptimal n_segments = {segment_values[best_idx]}  "
          f"(total = {totals[best_idx]:.2f} s)")


if __name__ == "__main__":
    print("======= Starting Segment Number Calibration =======")
    run_segment_calibration()
    print("\nSegment calibration completed!")
