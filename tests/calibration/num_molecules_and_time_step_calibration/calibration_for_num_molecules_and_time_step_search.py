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
    t_values = np.logspace(-4, 2, 100)  # From 0.0001 to 100 ms
    # total_sim_time = 100 # ms

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
        correction_term = (4 / (3 * d * np.sqrt(np.pi))) * S_over_V * np.sqrt(D0 * t)
        return D0 * (1 - correction_term)
    
    # Find the time where D(t) = 0 (validity limit)
    coefficient = (4 / (3 * d * np.sqrt(np.pi))) * S_over_V
    t_cutoff = (1 / coefficient)**2 / D0
    
    # Time range for short-time behavior - only up to where D(t) > 0
    max_valid_time = 0.95 * t_cutoff  # Use 95% of cutoff for safety
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
    
    print(f'Run {run_id}: Start setting up structures')
    if compartment == 'intra':
        sim.setup(structures=list(np.arange(0, sg3.nstructures)))  # seed inside structures
    print(f'Run {run_id}: Done setting up structures')
    
    # Validate compartment
    if compartment == 'intra':
        isInside = sim_util.validate_compartment_intra(sim, sg3)
        if not isInside.all():
            print(f"Run {run_id}: Warning: Not all molecules are inside structures")

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

def calculate_mae(D_numerical, D_long_interp, difftime, time_threshold=0.01):
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
                                          experiment_dir, time_threshold=0.01, a=0.5):
    '''Create and save comparison plot showing only 3 lines'''
    
    # Unpack analytical data
    t_long, D_long = analytical_data['longtime']
    t_short, D_short = analytical_data['shorttime']
    
    # Create the comparison plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot only the 3 required lines:
    # 1. D_numerical
    difftime_filtered = difftime[difftime >= time_threshold]
    D_numerical_filtered = D_numerical[difftime >= time_threshold]
    ax.semilogx(difftime_filtered, D_numerical_filtered, 'o', markerfacecolor='none', markeredgecolor='red', markersize=10, alpha=0.6, label=r'$D_{\perp, \mathrm{MC}}(t)$')
    
    # 2. D_analytical short time (only for t >= 0)
    t_short_positive = t_short[t_short >= 0]
    D_short_positive = D_short[:len(t_short_positive)]
    if len(t_short_positive) > 0:
        ax.semilogx(t_short_positive, D_short_positive, color='green', linestyle='dashed', linewidth=5,
                    label=r'$D_{\perp, t \rightarrow 0}(t) \approx D_0 \left(1 - \frac{4}{6\sqrt{\pi}} \frac{S}{V} \sqrt{D_0 t}\right)$', alpha=0.8)
    
    # 3. D_analytical long time
    ax.semilogx(t_long, D_long, 'b-', linewidth=5, label=r'$D_{\perp, \mathrm{GPA}}(t)$', alpha=0.8)
        
    # 4. Add D(t) = a²/(4t) asymptotic curve
    t_asymptotic = np.logspace(-1.5, 2, 100)  # From 0.01 to 100 ms
    D_asymptotic = a**2 / (4 * t_asymptotic)
    ax.semilogx(t_asymptotic, D_asymptotic, color='orange', linestyle='dashed', linewidth=5, 
            label=r'$D_{\perp, t \rightarrow \infty}(t) \approx a²/(4t)$', alpha=0.8)

    # Grid and labels
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_xlabel('Time (ms)', fontsize=20)
    ax.set_ylabel(r'$D_{\perp}(t)$ (μm²/ms)', fontsize=20)
    # ax.set_title(f'Analytical: D(t) vs Time\n'
    #             f'molecules: {molecules}, Time Step: {time_step}, Run: {run_id}, Radius: {a} μm', fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=17)
    # ax.tick_params(axis='both', which='minor', labelsize=25)
    # Set reasonable y-limits
    ax.set_ylim(0, D_short[0] * 1.1)
    
    # Create legend with error metrics
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_labels = ax.get_legend_handles_labels()[1]
    

    ax.legend(legend_elements, legend_labels, fontsize=18, framealpha=0.9)
    # plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tight_layout()
    
    # Save the figure
    plot_filename = f"cylinder_analysis_run_{run_id}.png"
    plot_filepath = os.path.join(experiment_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=500, bbox_inches='tight')
    plt.close()
    
    # print(f"Analysis plot saved: {plot_filepath}")
    return plot_filepath

def save_experiment_data(difftime, D_numerical, analytical_data, molecules, time_step, 
                        elapsed_time, run_id, base_dir, mae, 
                        time_threshold=0.01, a=0.5):
    '''Save experimental data in pickle format and create comparison plot'''
    
    # Create directory structure
    experiment_dir = os.path.join(base_dir, f"molecules_{int(molecules)}_timestep_{time_step}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create and save comparison plot
    plot_filepath = plot_numerical_vs_analytical_comparison(
        difftime, D_numerical, analytical_data, molecules, time_step, run_id, 
        elapsed_time, mae, experiment_dir, 
        time_threshold, a
    )
    
    # Prepare data to save
    experiment_data = {
        'molecules': molecules,
        'time_step': time_step,
        'run_id': run_id,
        'difftime': difftime,
        'D_numerical': D_numerical,
        'elapsed_time': elapsed_time,
        'analytical_data': analytical_data,
        'mae': mae,
        'time_threshold': time_threshold,
        'cylinder_radius': a,
        'plot_filepath': plot_filepath,
        'timestamp': time.time()
    }
    
    # Save as pickle file
    filename = f"cylinder_run_{run_id}.pkl"
    filepath = os.path.join(experiment_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(experiment_data, f)
    
    # print(f"Analysis data saved: {filepath}")
    return filepath

def run_validation_study():
    '''Run validation study for MCDS against analytical solution for cylinder'''
    
    # Parameters
    a = 0.5  # Cylinder radius um
    D0 = 2.0  # Initial diffusion coefficient um^2/ms
    time_threshold = 0.02  # Only calculate from 0.02 ms onwards
    compartment = 'intra'
    total_sim_time = 100

    # Study parameters
    # molecules_values = [int(1e6), int(5e5), int(2e5), int(1e5), int(5e4), int(2e4), int(1e4)]
    # time_step_values = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    # n_repeats = 10

    # molecules_values = [int(5e5), int(5e4), int(1e4)]
    # time_step_values = [0.002, 0.005, 0.01]
    # molecules_values = [int(1e3)]
    # time_step_values = [0.1]

    molecules_values = [int(5e4), int(1e4)]
    time_step_values = [0.005, 0.01]
    n_repeats = 2
    
    # Create results directory
    config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH = "./tests/calibration/num_molecules_and_time_step_calibration/figs/cylinder_validation_" + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) 
    if not os.path.exists(config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH):
        os.makedirs(config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH)
    
    # Calculate analytical solutions once
    print("Calculating analytical solutions for cylinder...")
    analytical_data = calculateDanalytical_single_cylinder(a, D0)
    
    # Create interpolation functions
    t_long, D_long = analytical_data['longtime']
    t_short, D_short = analytical_data['shorttime']
    

    D_long_interp = interp1d(np.log10(t_long), D_long, bounds_error=False, 
                            fill_value=(D_long[0], D_long[-1]))
    # Results storage
    all_results = []
    summary_results = []
    
    # Total combinations
    total_combinations = len(molecules_values) * len(time_step_values)
    current_combination = 0
    
    print(f"\nValidation study:")
    print(f"Simulation values calculated from t >= {time_threshold} ms onwards")
    
    # Run experiments
    for molecules in molecules_values:
        for time_step in time_step_values:
            current_combination += 1
            print(f"\n{'='*60}")
            print(f"Combination {current_combination}/{total_combinations}")
            print(f"molecules: {molecules}, Time Step: {time_step}")
            print(f"{'='*60}")
            
            mae_values = []
            computation_times = []

            # Run multiple repetitions
            for run_id in range(n_repeats):
                print(f"\nRun {run_id + 1}/{n_repeats}")
                
                start_time = time.time()
                
                try:
                    # Run simulation
                    difftime, D_numerical = calculateDnumerical(config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH, a, D0, molecules, time_step, compartment, total_sim_time, run_id)
                    elapsed_time = time.time() - start_time
                    
                    # Calculate MAE using only long-time analytical for t >= threshold
                    mae = calculate_mae(D_numerical, D_long_interp, difftime, time_threshold)
                    
                    # Record computation time
                    computation_times.append(elapsed_time)
                    mae_values.append(mae)
                    
                    # Save experiment data with plot
                    save_experiment_data(difftime, D_numerical, analytical_data, molecules, time_step, 
                                       elapsed_time, run_id, config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH, mae, 
                                       time_threshold, a)
                    
                    # Store individual result
                    result = {
                        'molecules': molecules,
                        'time_step': time_step,
                        'run_id': run_id,
                        'mae': mae,
                        'computation_time': elapsed_time
                    }
                    all_results.append(result)
                    
                    print(f"Run {run_id}: MAE = {mae:.4f}, Time = {elapsed_time:.2f}s")
                    
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
                    
                    print(f"\nSummary for molecules={molecules}, time_step={time_step}:")
                    print(f"MAE: {summary['mae']:.4f} ± {summary['std_mae']:.4f}")
                    print(f"Mean Time: {summary['mean_computation_time']:.2f}s ± {summary['std_computation_time']:.2f}s")
    
    # Save summary results
    save_summary_results(summary_results, config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH, a, D0, time_threshold)
    
    # Create analysis plots
    create_analysis_plots(summary_results, config_params.NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH, a, D0, time_threshold)
    
    # Find optimal parameters
    find_optimal_parameters(summary_results)
    
    return all_results, summary_results

def save_summary_results(summary_results, base_dir, a, D0, time_threshold):
    '''Save summary results to CSV'''
    
    df = pd.DataFrame(summary_results)
    
    # Save detailed summary
    csv_path = os.path.join(base_dir, f"cylinder_validation_summary_a{a}_D0{D0}_threshold{time_threshold}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary results saved to: {csv_path}")
    print(f"\nTop 10 combinations by MAE:")
    if not df.empty:
        print(df.nsmallest(10, 'mae')[['molecules', 'time_step', 'mae', 'mean_computation_time']].to_string(index=False))

def create_analysis_plots(summary_results, base_dir, a, D0, time_threshold):
    '''Create comprehensive analysis plots'''
    
    df = pd.DataFrame(summary_results)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: MAE vs Molecules for different time steps
    time_steps = df['time_step'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_steps)))
    
    for i, ts in enumerate(sorted(time_steps)):
        data = df[df['time_step'] == ts]
        ax1.loglog(data['molecules'], data['mae'], 'o-', 
                  color=colors[i], label=f'dt={ts}', linewidth=2, markersize=6)
        ax1.fill_between(data['molecules'], 
                        data['mae'] - data['std_mae'],
                        data['mae'] + data['std_mae'],
                        color=colors[i], alpha=0.2)
    
    ax1.set_xlabel('Number of Molecules', fontsize=17)
    ax1.set_ylabel(f'MAE', fontsize=17)
    ax1.set_title(f'MAE vs Number of Molecules', fontsize=17)
    # 2-column legend for plot 1
    ax1.legend(ncol=4, fontsize=13)
    ax1.grid(True, alpha=0.5)
    # Show axis origins
    ax1.set_xlim(left=df['molecules'].min() * 0.8)
    ax1.set_ylim(bottom=df['mae'].min() * 0.8, top=(df['mae']+df['std_mae']).max() * 1.3)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.tick_params(axis='both', which='minor', labelsize=12)
    
    # Plot 2: MAE vs Time Step for different molecules
    molecules_list = df['molecules'].unique()
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(molecules_list)))
    
    for i, mol in enumerate(sorted(molecules_list)):
        data = df[df['molecules'] == mol]
        ax2.loglog(data['time_step'], data['mae'], 's-', 
                  color=colors2[i], label=f'{mol:.0e} molecules', linewidth=2, markersize=6)
        ax2.fill_between(data['time_step'], 
                        data['mae'] - data['std_mae'],
                        data['mae'] + data['std_mae'],
                        color=colors2[i], alpha=0.2)
    
    ax2.set_xlabel('Time Step (ms)', fontsize=17)
    ax2.set_ylabel(f'MAE', fontsize=17)
    ax2.set_title(f'MAE vs Time Step', fontsize=17)
    # 2-column legend for plot 2
    ax2.legend(ncol=2, fontsize=13)
    ax2.grid(True, alpha=0.5)
    # Show axis origins
    ax2.set_xlim(left=df['time_step'].min() * 0.8)
    ax2.set_ylim(bottom=df['mae'].min() * 0.8)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='minor', labelsize=12)
    
    # Plot 3: Computation Time vs Molecules
    for i, ts in enumerate(sorted(time_steps)):
        data = df[df['time_step'] == ts]
        ax3.loglog(data['molecules'], data['mean_computation_time'], 'o-', 
                  color=colors[i], label=f'dt={ts}', linewidth=2, markersize=6)
    
    ax3.set_xlabel('Number of Molecules', fontsize=17)
    ax3.set_ylabel('Mean Computation Time (s)', fontsize=17)
    ax3.set_title('Computation Time vs Number of Molecules', fontsize=17)
    # 2-column legend for plot 3
    ax3.legend(ncol=2, fontsize=13)
    ax3.grid(True, alpha=0.5)
    # Show axis origins
    ax3.set_xlim(left=df['molecules'].min() * 0.8)
    ax3.set_ylim(bottom=df['mean_computation_time'].min() * 0.8)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.tick_params(axis='both', which='minor', labelsize=12)
    
    # Plot 4: BAR PLOT - MAE vs Number of Molecules with Time Steps as grouped bars
    molecules_sorted = sorted(df['molecules'].unique())
    time_steps_sorted = sorted(df['time_step'].unique())
    
    # Set up bar positions
    n_molecules = len(molecules_sorted)
    n_time_steps = len(time_steps_sorted)
    bar_width = 0.8 / n_time_steps
    x_pos = np.arange(n_molecules)
    
    # Colors for different time steps
    colors_bar = plt.cm.Set3(np.linspace(0, 1, n_time_steps))
    
    # Create bars for each time step
    for i, time_step in enumerate(time_steps_sorted):
        means = []
        stds = []
        
        for molecules in molecules_sorted:
            data_point = df[(df['molecules'] == molecules) & (df['time_step'] == time_step)]
            if not data_point.empty:
                means.append(data_point['mae'].iloc[0])
                stds.append(data_point['std_mae'].iloc[0])
            else:
                means.append(0)
                stds.append(0)
        
        # Plot bars with error bars
        bar_positions = x_pos + i * bar_width - (n_time_steps - 1) * bar_width / 2
        bars = ax4.bar(bar_positions, means, bar_width, 
                      yerr=stds, capsize=3,
                      color=colors_bar[i], alpha=0.8,
                      label=f'dt = {time_step}',
                      edgecolor='black', linewidth=0.5)
    
    # Customize the bar plot
    ax4.set_xlabel('Number of Molecules', fontsize=17)
    ax4.set_ylabel('MAE', fontsize=17)
    ax4.set_title('MAE vs Number of Molecules by Time Step', fontsize=17)
    
    # Set x-axis labels
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{int(mol):.0e}' for mol in molecules_sorted])
    
    # Add legend
    ax4.legend(ncol=4, fontsize=13, loc='upper right')
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='both', which='major', labelsize=15)
    ax4.tick_params(axis='both', which='minor', labelsize=12)
    
    # Set y-axis to start from 0 for better bar plot visualization
    ax4.set_ylim(bottom=0.0)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(base_dir, f"analytical_validation_analysis_a{a}_D0{D0}_threshold{time_threshold}.png")    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"Analysis plots saved to: {plot_path}")

def find_optimal_parameters(summary_results):
    '''Find optimal parameters based on MAE'''
    
    df = pd.DataFrame(summary_results)
    # Find best combination overall
    if not df.empty:
        best_idx = df['mae'].idxmin()
        best_result = df.loc[best_idx]
        
        print(f"\n{'='*60}")
        print("OPTIMAL PARAMETERS for Analytical")
        print(f"{'='*60}")
        print(f"Best molecules: {best_result['molecules']}")
        print(f"Best Time Step: {best_result['time_step']}")
        print(f"MAE: {best_result['mae']:.4f} ± {best_result['std_mae']:.4f}")
        print(f"Mean Computation Time: {best_result['mean_computation_time']:.2f}s ± {best_result['std_computation_time']:.2f}s")

if __name__ == "__main__":
    print("=======Starting Analytical Validation Study=======")
    
    # Run the validation study
    all_results, summary_results = run_validation_study()
    
    print("\nAnalytical validation study completed!")