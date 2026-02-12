import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from scipy.special import jnp_zeros
import os
import pycuda.gpuarray as gpuarray
import numpy as np
import logging

# Suppress matplotlib font substitution messages
logging.getLogger('matplotlib.mathtext').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
import simulation_toolkit.simulation_engine.diffsim3d as ds3
import simulation_toolkit.simulation_engine.geometry as geom

import simulation_toolkit.simulation_engine.helper.simulation_report as simrep
import simulation_toolkit.simulation_engine.helper.sim_util as sim_util
import simulation_toolkit.utils.common_utils as common_util

import simulation_toolkit.toolkit_params as config_params

import simulation_toolkit.utils.adjust_geometry as ag
from scipy.interpolate import interp1d
from matplotlib.pyplot import cm
import time
import os
  
def calculateDanalytical_longtime(a, D0):
    '''Analytical intra-axonal radial D(t) at from 
    Burcaw 2015 https://doi.org/10.1016/j.neuroimage.2015.03.061
    derived from Stepišnik (1993) and Callaghan (1995)'''
    
    # Time range (normalized units)
    t_values = np.logspace(-4, 2, 100)  # From 0.0001 to 100 ms (total_sim_time)

    # Get the first few roots of the derivative of J1(x)
    beta_1k = jnp_zeros(1, 100)  # Get first 100 roots
    
    def D_t(t):
        # First term
        term1 = a**2 / (4*t)
        
        # Sum term
        sum_term = 0
        for k in range(100):  # Use first 100 terms of infinite sum
            beta = beta_1k[k]
            sum_term += np.exp(-beta**2 * D0 * t / a**2) / (beta**2 * (beta**2 - 1))
        
        return term1 - (2*a**2/t) * sum_term
    
    # Calculate D(t) for all time values
    D_values = [D_t(t) for t in t_values]
    
    return t_values, D_values
  
def calculateDanalytical_shorttime(a, D0, d=2):
    '''Analytical diffusion coefficient for short-time limit
    D(t) ≈ D0 * (1 - 4/(3d√π) * (S/V) * √(D0*t))
    For cylinder: S/V = 2/a (surface to volume ratio)
    d = 2 for radial diffusion in 2D plane
    Only valid while D(t) > 0'''
    
    # Surface to volume ratio for cylinder (2D radial)
    S_over_V = 2 / a
    
    def D_t_shorttime(t):
        # Short-time expression
        surface_to_volume_ratio_term = (4 / (3 * d * np.sqrt(np.pi))) * S_over_V * np.sqrt(D0 * t)
        return D0 * (1 - surface_to_volume_ratio_term)
    
    # Find the time where D(t) = 0 (validity limit)
    coefficient = (4 / (3 * d * np.sqrt(np.pi))) * S_over_V
    t_cutoff = (1 / coefficient)**2 / D0
    
    # Time range for short-time behavior - only up to where D(t) > 0
    max_valid_time = 0.95 * t_cutoff
    t_values = np.logspace(-4, np.log10(max_valid_time), 100)
    
    # Calculate D(t) for all time values
    D_values = [D_t_shorttime(t) for t in t_values]
    
    return t_values, D_values
  
def calculateDanalytical_single_cylinder(a, D0):
    '''Calculate analytical D(t) for a single cylinder
    Returns both long-time and short-time solutions'''
    t_long, D_long = calculateDanalytical_longtime(a, D0)
    t_short, D_short = calculateDanalytical_shorttime(a, D0)
    
    return {
        'longtime': (t_long, D_long),
        'shorttime': (t_short, D_short)
    }

def calculateDnumerical(base_dir, a, D0, molecules, time_step, compartment, total_sim_time, run_id=0):
    '''Calculate numerical D(t) for 1 cylinder'''
    # Simulation box dimensions
    Lx = 20.0  # um
    Ly = 20.0  # um  
    Lz = 20.0  # um
    
    D = D0  # um^2/ms
    T2 = 100  # ms
    rho = 1  # fractional water density
    
    sg3 = geom.SimGeometry3D(Lx, Ly, Lz, D, T2, rho)
    
    # Create single cylinder along z-axis (centered at origin)
    nsphere = 100  # Number of cylinder segments along z-axis
    
    # Create cylinder segments along z-axis
    sz = np.linspace(-Lz/2, Lz/2, nsphere)
    sy = np.full(sz.shape, 0)  # Center at y=0
    sx = np.full(sz.shape, 0)  # Center at x=0
    sr = np.full(sz.shape, a)  # Radius = a
    
    # Add periodic boundaries in z-direction
    sz1 = np.concatenate((sz[-30:]-Lz, sz, sz[:30]+Lz), axis=0)
    sy1 = np.concatenate((sy[-30:], sy, sy[:30]), axis=0)
    sx1 = np.concatenate((sx[-30:], sx, sx[:30]), axis=0)
    sr1 = np.concatenate((sr[-30:], sr, sr[:30]), axis=0)
    
    spstruc = geom.Structure3D(sx1, sy1, sz1, sr1, D, T2, rho)
    sg3.add_structure(spstruc)

    total_sim_time = 100 # ms
    dt = time_step  # time step in ms
    nt = int(total_sim_time / dt)  # total number of steps thru time
    
    sim = ds3.DiffSim3d(sg3, int(molecules))
    nsegx, nsegy, nsegz = 5, 5, 5
    sim.set_segments(nsegx=nsegx, nsegy=nsegy, nsegz=nsegz)

    if compartment == 'intra':
        sim.setup(structures=list(np.arange(0, sg3.nstructures)))  # seed inside structures
    
    numsteps = nt + 1  # +1 for initial condition
    Dxarray, Dyarray, Dzarray = gpuarray.zeros(numsteps, dtype=np.float32), gpuarray.zeros(numsteps, dtype=np.float32), gpuarray.zeros(numsteps, dtype=np.float32)
    # Arrays for storing second (variance) and fourth moments (NOT used but allocated due to simulation function signature)
    Kx2array, Ky2array, Kz2array = gpuarray.zeros(numsteps, dtype=np.float32), gpuarray.zeros(numsteps, dtype=np.float32), gpuarray.zeros(numsteps, dtype=np.float32)
    Kx4array, Ky4array, Kz4array = gpuarray.zeros(numsteps, dtype=np.float32), gpuarray.zeros(numsteps, dtype=np.float32), gpuarray.zeros(numsteps, dtype=np.float32)
    
    # Pre-allocate CPU result arrays
    Dxstep, Dystep, Dzstep, difftime = np.zeros(numsteps), np.zeros(numsteps), np.zeros(numsteps), np.zeros(numsteps)
    
    # Initial values
    Dxstep[0], Dystep[0], Dzstep[0], difftime[0] = D, D, D, 0
    
    current_time = 0.0
    step_idx = 1
    
    # Simulation loop
    while current_time < nt*dt:
        # Take time step
        sim.step(time_step)
        current_time += time_step
        
        # Use the GPU kernel to compute displacements
        sim.calculate_diffusion_coefficients_and_kurtoses(step_idx, current_time, 
                                    Dxarray, Dyarray, Dzarray,
                                    Kx2array, Ky2array, Kz2array, 
                                    Kx4array, Ky4array, Kz4array)
        difftime[step_idx-1]=current_time
        step_idx += 1

    # After the loop, copy results back to CPU once
    Dxstep, Dystep, Dzstep, difftime = np.array(Dxarray.get())[1:], np.array(Dyarray.get())[1:], np.array(Dzarray.get())[1:], np.array(difftime)[1:]
    
    # Return radial diffusion coefficient (average of x and y)
    D_numerical = (Dxstep + Dystep) / 2
    
    return difftime, D_numerical

def calculate_mae(D_numerical, D_long_interp, difftime, time_threshold=0.05):
    '''Calculate combined MAE using long-time analytical for all comparisons when t >= 10^-2 ms'''
    
    # Filter data to only use values from 10^-2 ms onwards
    valid_mask = difftime >= time_threshold
    
    if not np.any(valid_mask):
        print("No valid time points found for error calculation")
        return np.nan
    
    # Use only filtered data
    difftime_filtered = difftime[valid_mask]
    D_numerical_filtered = D_numerical[valid_mask]
    
    # Use long-time analytical for all comparisons
    D_analytical_filtered = D_long_interp(np.log10(difftime_filtered))
    
    # Calculate relative error
    absolute_error = np.abs(D_numerical_filtered - D_analytical_filtered)
    
    mae = np.mean(absolute_error)  # as percentage
    
    return mae

def plot_numerical_vs_analytical_comparison(difftime, D_numerical, analytical_data, 
                                          molecules, time_step, run_id, elapsed_time, 
                                          percentage_mae,
                                          experiment_dir, time_threshold=0.05, a=0.5):
    '''Create and save comparison plot showing only 3 lines'''
    
    # Unpack analytical data
    t_long, D_long = analytical_data['longtime']
    t_short, D_short = analytical_data['shorttime']
    
    # Create the comparison plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot 3 lines:
    # 1. D_numerical
    difftime_filtered = difftime[difftime >= time_threshold]
    D_numerical_filtered = D_numerical[difftime >= time_threshold]
    ax.semilogx(difftime_filtered, D_numerical_filtered, 'o', markerfacecolor='none', markeredgecolor='red', markersize=10, alpha=0.6, label=r'$D_{\perp, \mathrm{MC}}(t)$')
    
    # 2. D_analytical short time
    t_short_positive = t_short[t_short >= 0]
    D_short_positive = D_short[:len(t_short_positive)]
    if len(t_short_positive) > 0:
        ax.semilogx(t_short_positive, D_short_positive, color='green', linestyle='dashed', linewidth=5,
                    label=r'$D_{\perp, t \rightarrow 0}(t) $', alpha=0.8)
    
    # 3. D_analytical long time
    ax.semilogx(t_long, D_long, 'b-', linewidth=5, label=r'$D_{\perp, \mathrm{GPA}}(t)$', alpha=0.8)
        
    # 4. Add D(t) = a²/(4t) asymptotic curve
    t_asymptotic = np.logspace(-1.5, 2, 100)  # From 0.01 to 100 ms
    D_asymptotic = a**2 / (4 * t_asymptotic)
    ax.semilogx(t_asymptotic, D_asymptotic, color='orange', linestyle='dashed', linewidth=5, 
            label=r'$D_{\perp, t \rightarrow \infty}(t) \approx a²/(4t)$', alpha=0.8)

    # Grid and labels
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_xlabel('Time (ms)', fontsize=25)
    ax.set_ylabel(r'$D_{\perp}(t)$ (μm²/ms)', fontsize=25)
    # ax.set_title(f'Analytical: D(t) vs Time\n'
    #             f'molecules: {molecules}, Time Step: {time_step}, Radius: {a} μm', fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.set_ylim(0, D_short[0] * 1.1)
    
    # Create legend with error metrics
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_labels = ax.get_legend_handles_labels()[1]
    
    ax.legend(legend_elements, legend_labels, fontsize=25, framealpha=0.9)
    plt.tight_layout()
    
    # Save the figure
    plot_filename = f"cylinder_analysis_run_{run_id}.png"
    plot_filepath = os.path.join(experiment_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=500, bbox_inches='tight')
    plt.close()
    return

def save_experiment_data(difftime, D_numerical, analytical_data, molecules, time_step, 
                        elapsed_time, run_id, base_dir, mae, 
                        time_threshold=0.05, a=0.5):
    '''Save comparison plot from experiment'''
    # Create directory structure
    experiment_dir = os.path.join(base_dir, f"molecules_{int(molecules)}_timestep_{time_step}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create and save comparison plot
    plot_numerical_vs_analytical_comparison(
        difftime, D_numerical, analytical_data, molecules, time_step, run_id, 
        elapsed_time, mae, experiment_dir, 
        time_threshold, a
    )
    return
  
def run_validation_study():
    '''Run validation study for MCDS against analytical solution for cylinder'''
    
    # Parameters
    a = 0.5  # Cylinder radius um
    D0 = 2.0  # Initial diffusion coefficient um^2/ms
    time_threshold = 0.03  # For calculating MAE
    compartment = 'intra'
    total_sim_time = 100

    # Study parameters
    molecules_values = [int(1e6), int(5e5), int(2e5), int(1e5), int(5e4), int(2e4), int(1e4)]
    time_step_values = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    n_repeats = 5

    # molecules_values = [int(5e4), int(2e4), int(1e4)]
    # time_step_values = [0.005, 0.01]
    # n_repeats = 1
    
    # Create results directory
    config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH = "./tests/calibration/num_molecules_and_time_step_calibration/figs/cylinder_validation_" + str(datetime.now().strftime("%Y-%m-%d_%H-%M")) 
    if not os.path.exists(config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH):
        os.makedirs(config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH)
    
    # Calculate analytical solution
    print("Calculating analytical solutions for cylinder...")
    analytical_data = calculateDanalytical_single_cylinder(a, D0)
    
    # Create interpolation function
    t_long, D_long = analytical_data['longtime']
    D_long_interp = interp1d(np.log10(t_long), D_long, bounds_error=False, 
                            fill_value=(D_long[0], D_long[-1]))
    # Results storage
    summary_results = []
    
    # Total combinations
    total_combinations = len(molecules_values) * len(time_step_values)
    current_combination = 0
    
    # Run different combinations 
    for molecules in molecules_values:
        for time_step in time_step_values:
            current_combination += 1
            print(f"{'='*35}")
            print(f"Combination {current_combination}/{total_combinations}")
            print(f"Molecules: {molecules}, Time step: {time_step}")
            
            mae_values = []
            computation_times = []

            # Run multiple repetitions
            for run_id in range(n_repeats):
                print(f"Run {run_id + 1}/{n_repeats}")
                
                start_time = time.time()
                
                try:
                    # Run simulation
                    difftime, D_numerical = calculateDnumerical(config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH, a, D0, molecules, time_step, compartment, total_sim_time, run_id)
                    elapsed_time = time.time() - start_time
                    
                    # Calculate MAE between analytical and numerical results
                    mae = calculate_mae(D_numerical, D_long_interp, difftime, time_threshold)
                    
                    # Record computation time
                    computation_times.append(elapsed_time)
                    mae_values.append(mae)
                    
                    # Save experiment data with plot
                    save_experiment_data(difftime, D_numerical, analytical_data, molecules, time_step, 
                                       elapsed_time, run_id, config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH, mae, 
                                       time_threshold, a)                    
                except Exception as e:
                    print(f"Error in run {run_id}: {str(e)}")
                    continue
            
            # Calculate summary statistics for this combination
            if mae_values:
                # Filter out NaN values
                valid_mae = [x for x in mae_values if not np.isnan(x)]
                
                if valid_mae:
                    summary = {
                        'molecules': molecules,
                        'time_step': time_step,
                        'time_threshold': time_threshold,
                        'mae': np.mean(valid_mae),
                        'std_mae': np.std(valid_mae),
                        'min_mae': np.min(valid_mae),
                        'max_mae': np.max(valid_mae),
                        'mean_computation_time': np.mean(computation_times),
                        'std_computation_time': np.std(computation_times),
                        'n_successful_runs': len(valid_mae)
                    }
                    summary_results.append(summary)
                    
    # Save summary results
    save_summary_results(summary_results, config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH, a, D0, time_threshold)
    
    # Create analysis plots
    create_analysis_plots(summary_results, config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH, a, D0, time_threshold)
    return

def save_summary_results(summary_results, base_dir, a, D0, time_threshold):
    '''Save summary results to CSV'''  
    df = pd.DataFrame(summary_results)
    csv_path = os.path.join(base_dir, f"cylinder_validation_summary_a{a}_D0{D0}_threshold{time_threshold}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary results saved to: {csv_path}")
    
def create_analysis_plots(summary_results, base_dir, a, D0, time_threshold):
    '''Create analysis plots'''
    df = pd.DataFrame(summary_results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: MAE vs Time Step for different molecules
    molecules_list = df['molecules'].unique()
    colors = plt.cm.plasma(np.linspace(0, 1, len(molecules_list)))
    
    for i, mol in enumerate(sorted(molecules_list)):
        data = df[df['molecules'] == mol]
        ax1.loglog(data['time_step'], data['mae'], 's-', 
                  color=colors[i], label=f'{mol:.0e} molecules', linewidth=2, markersize=6)
        ax1.fill_between(data['time_step'], 
                        data['mae'] - data['std_mae'],
                        data['mae'] + data['std_mae'],
                        color=colors[i], alpha=0.2)
    
    ax1.set_xlabel('Time Step (ms)', fontsize=17)
    ax1.set_ylabel(f'MAE', fontsize=17)
    # ax1.set_title(f'MAE vs Time Step', fontsize=17)
    ax1.legend(ncol=2, fontsize=13)
    ax1.grid(True, alpha=0.5)
    ax1.set_xlim(left=df['time_step'].min() * 0.8)
    ax1.set_ylim(bottom=df['mae'].min() * 0.8)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.tick_params(axis='both', which='minor', labelsize=12)
    
    # Plot 2: Computation Time vs Molecules
    time_steps = df['time_step'].unique()
    colors2 = plt.cm.viridis(np.linspace(0, 1, len(time_steps)))
    
    for i, ts in enumerate(sorted(time_steps)):
        data = df[df['time_step'] == ts]
        ax2.loglog(data['molecules'], data['mean_computation_time'], 'o-', 
                  color=colors2[i], label=f'dt={ts}', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Number of Molecules', fontsize=17)
    ax2.set_ylabel('Computation Time (sec)', fontsize=17)
    # ax2.set_title('Computation Time vs Number of Molecules', fontsize=17)
    ax2.legend(ncol=2, fontsize=13)
    ax2.grid(True, alpha=0.5)
    ax2.set_xlim(left=df['molecules'].min() * 0.8)
    ax2.set_ylim(bottom=df['mean_computation_time'].min() * 0.8)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='minor', labelsize=12)    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(base_dir, f"analytical_validation_analysis_a{a}_D0{D0}_threshold{time_threshold}.png")    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"\nAnalysis plots saved to: {plot_path}")

if __name__ == "__main__":
    print("=======Starting Analytical Validation Study=======")
    run_validation_study()
    print("\nAnalytical validation study completed!")