import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
import sys
from pathlib import Path
import logging

# Suppress matplotlib mathtext INFO messages
logging.getLogger('matplotlib.mathtext').setLevel(logging.WARNING)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from simulation_toolkit.utils.metric_visualization.core import gather_experiment_files, extract_diameter_from_filename

def load_ADC_data_pickle(file_path):
    """Load pickle file and return ADC data."""
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def average_multi_runs_data_by_diameter(file_paths):
    """
    Calculate mean and std across multiple runs for each diameter.
    
    Args:
        file_paths: List of file paths for the same K value and compartment
    
    Returns:
        Dictionary with diameter as key and averaged data as values
    """    
    # Group files by diameter
    grouped_data = defaultdict(list)
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if filename.endswith('.pkl'):
            diameter = extract_diameter_from_filename(filename)
            
            if diameter:
                try:
                    Dx_Dy_Dz_difftime = load_ADC_data_pickle(file_path)
                    grouped_data[diameter].append(Dx_Dy_Dz_difftime)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    # Calculate averages for each diameter
    averaged_results = {}
    
    # BOTH INTRA & EXTRA averaged together
    # exit(0)
    for diameter, multi_runs_data in grouped_data.items():
        if not multi_runs_data:
            continue
            
        # Collect all time series for averaging
        all_dxy = []
        all_time = []
        
        for Dx_Dy_Dz_difftime in multi_runs_data:
            Dx, Dy, Dz, diff_time = Dx_Dy_Dz_difftime[:,0], Dx_Dy_Dz_difftime[:,1], Dx_Dy_Dz_difftime[:,2], Dx_Dy_Dz_difftime[:,3]
            Dxy = (Dx + Dy) / 2  # Radial diffusion coefficient
            all_dxy.append(Dxy)
            all_time.append(diff_time)
        
        # Convert to numpy arrays
        
        all_dxy = np.array(all_dxy)
        all_time = np.array(all_time)
        
        # Calculate mean and standard deviation
        mean_dxy = np.mean(all_dxy, axis=0)
        std_dxy = np.std(all_dxy, axis=0, ddof=1)
        mean_time = np.mean(all_time, axis=0)
        
        averaged_results[diameter] = {
            'Dxy_mean': mean_dxy,
            'Dxy_std': std_dxy,
            'diff_time_mean': mean_time,
            'num_runs': len(multi_runs_data)
        }
    
    return averaged_results

def plot_Dxy_across_diameters_for_K(organized_files, k_value, compartment, output_dir, diff_time_limit=120):
    """
    Plot Dxy across different diameters for a specific K value and compartment.
    
    Args:
        organized_files: Dictionary from gather_experiment_files()
        k_value: K value to plot (e.g., 'K10')
        compartment: 'intra' or 'extra'
        output_dir: Directory to save plots
        diff_time_limit: Maximum diffusion time to plot
    """
    
    # Get files for this K value and compartment
    if k_value not in organized_files:
        print(f"No data found for {k_value}")
        return
    
    if compartment not in organized_files[k_value]:
        print(f"No {compartment} data found for {k_value}")
        return
    
    # Get all file paths for this K and compartment
    all_files = []
    for diameter, file_paths in organized_files[k_value][compartment].items():
        all_files.extend(file_paths)
    
    if not all_files:
        print(f"No files found for {k_value} {compartment}")
        return
    
    # Average data by diameter
    averaged_data = average_multi_runs_data_by_diameter(all_files)
    
    if not averaged_data:
        print(f"No valid data to plot for {k_value} {compartment}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Colors for different diameters
    colors = {'1.68': 'red', '2.58': 'blue', '3.5': 'green', '4.5': 'purple', '2.58': 'blue', '3.5': 'green'}
    
    # Plot each diameter
    for diameter, data in sorted(averaged_data.items(), key=lambda x: float(x[0])):
        diff_time_mean = data['diff_time_mean']
        dxy_mean = data['Dxy_mean']
        dxy_std = data['Dxy_std']
        
        # Filter by time limit
        time_mask = diff_time_mean <= diff_time_limit
        diff_time_filtered = diff_time_mean[time_mask]
        dxy_mean_filtered = dxy_mean[time_mask]
        dxy_std_filtered = dxy_std[time_mask]
        
        color = colors.get(diameter, 'black')
        
        # Plot mean line
        ax.semilogx(diff_time_filtered, dxy_mean_filtered, 
                   color=color, label=str(diameter)+' µm', linewidth=3)
        
        # Add shaded error region
        ax.fill_between(diff_time_filtered,
                       dxy_mean_filtered - dxy_std_filtered,
                       dxy_mean_filtered + dxy_std_filtered,
                       color=color, alpha=0.3)
    
    # Formatting
    ax.set_xlabel(r'$t\;(\mathrm{ms})$', fontsize=25)
    if compartment == 'intra':
        ax.set_ylabel(r'$D_{\mathrm{i},\!\!\perp}\;(\mathrm{\mu m}^2/\mathrm{ms})$', fontsize=25)
    else:
        ax.set_ylabel(r'$D_{\mathrm{e},\!\!\perp}\;(\mathrm{\mu m}^2/\mathrm{ms})$', fontsize=25)
    
    kappa_num = k_value[1:]  # Remove 'K' prefix
    ax.set_title(f"$\\kappa$={kappa_num}", fontsize=20)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.002, diff_time_limit])
    ax.set_ylim([0.0, 3.0])
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"Dperp_across_diameter_{k_value}_{compartment}.png"
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=500, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {plot_filepath}")

def main():
    """Main function to generate plots."""
    
    # Gather all experiment files
    base_folder = "./experiment/visualization"
    print("Gathering experiment files...")
    organized_files = gather_experiment_files(base_folder)
    
    # Output directory
    output_dir = "./experiment/visualization/plots/Dperp_across_diameter"
    
    # Get all available K values
    k_values = list(organized_files.keys())
    compartments = ['intra', 'extra']
    
    print(f"Found K values: {k_values}")
    
    # Generate plots for each K value and compartment
    for k_value in k_values:
        for compartment in compartments:
            print(f"\nProcessing {k_value} {compartment}...")
            plot_Dxy_across_diameters_for_K(
                organized_files, k_value, compartment, output_dir
            )
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()