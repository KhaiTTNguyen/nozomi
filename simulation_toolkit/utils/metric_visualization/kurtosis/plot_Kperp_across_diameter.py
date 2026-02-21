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

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from simulation_toolkit.utils.metric_visualization.core import gather_experiment_files, extract_diameter_from_filename

def load_kurtosis_data_pickle(file_path):
    """Load pickle file and return kurtosis data."""
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def average_multi_runs_kurtosis_by_diameter(file_paths):
    """
    Calculate mean and std across multiple runs for each diameter.
    
    Args:
        file_paths: List of file paths for the same K value and compartment
    
    Returns:
        Dictionary with diameter as key and averaged kurtosis data as values
    """    
    # Group files by diameter
    grouped_data = defaultdict(list)
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if filename.endswith('.pkl'):
            diameter = extract_diameter_from_filename(filename)
            
            if diameter:
                try:
                    Kx_Ky_Kz_difftime = load_kurtosis_data_pickle(file_path)
                    grouped_data[diameter].append(Kx_Ky_Kz_difftime)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    # Calculate averages for each diameter
    averaged_results = {}

    for diameter, multi_runs_data in grouped_data.items():
        if not multi_runs_data:
            continue
            
        # Collect all time series for averaging
        all_kxy = []
        all_time = []
        
        for Kx_Ky_Kz_difftime in multi_runs_data:
            Kx, Ky, Kz, diff_time = Kx_Ky_Kz_difftime[:,0], Kx_Ky_Kz_difftime[:,1], Kx_Ky_Kz_difftime[:,2], Kx_Ky_Kz_difftime[:,3]
            Kxy = (Kx + Ky) / 2  # Radial kurtosis coefficient
            all_kxy.append(Kxy)
            all_time.append(diff_time)
        
        # Convert to numpy arrays
        all_kxy = np.array(all_kxy)
        all_time = np.array(all_time)
        
        # Calculate mean and standard deviation
        mean_kxy = np.mean(all_kxy, axis=0)
        std_kxy = np.std(all_kxy, axis=0, ddof=1)
        mean_time = np.mean(all_time, axis=0)
        
        averaged_results[diameter] = {
            'Kxy_mean': mean_kxy,
            'Kxy_std': std_kxy,
            'diff_time_mean': mean_time,
            'num_runs': len(multi_runs_data)
        }
    
    return averaged_results

def plot_Kxy_across_diameters_for_K(organized_files, k_value, compartment, output_dir, diff_time_limit=120):
    """
    Plot Kxy (perpendicular kurtosis) across different diameters for a specific K value and compartment.
    
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
    averaged_data = average_multi_runs_kurtosis_by_diameter(all_files)
    
    if not averaged_data:
        print(f"No valid data to plot for {k_value} {compartment}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10,5))
    
    # Colors for different diameters
    colors = {'1.68': 'red', '2.58': 'blue', '3.5': 'green', '4.5': 'purple', '2.58': 'blue', '3.5': 'green'}
    
    # Plot each diameter
    for diameter, data in sorted(averaged_data.items(), key=lambda x: float(x[0])):
        diff_time_mean = data['diff_time_mean']
        kxy_mean = data['Kxy_mean']
        kxy_std = data['Kxy_std']
        
        # Filter by time limit
        time_mask = diff_time_mean <= diff_time_limit
        diff_time_filtered = diff_time_mean[time_mask]
        kxy_mean_filtered = kxy_mean[time_mask]
        kxy_std_filtered = kxy_std[time_mask]
        
        color = colors.get(diameter, 'black')
        
        # Plot mean line
        ax.plot(diff_time_filtered, kxy_mean_filtered, 
                   color=color, label=str(diameter)+' µm', linewidth=3)
        
        # Add shaded error region
        ax.fill_between(diff_time_filtered,
                       kxy_mean_filtered - kxy_std_filtered,
                       kxy_mean_filtered + kxy_std_filtered,
                       color=color, alpha=0.3)
    
    # Formatting
    ax.set_xlabel(r'$t\;(\mathrm{ms})$', fontsize=27)
    if compartment == 'intra':
        ax.set_ylabel(r'$K_{\mathrm{i},\!\!\perp}$', fontsize=27)
        ax.set_xlim([0.002, 100])
        ax.set_ylim([0.0, 10])
    else:
        ax.set_ylabel(r'$K_{\mathrm{e},\!\!\perp}$', fontsize=27)
        ax.set_xlim([0.002, 20])
        ax.set_ylim([0.0, 0.8])
    
    kappa_num = k_value[1:]  # Remove 'K' prefix
    ax.set_title(f"$\\kappa$={kappa_num}", fontsize=27)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"Kperp_across_diameter_{k_value}_{compartment}.png"
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=500, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {plot_filepath}")

def main():
    """Main function to generate plots."""
    
    # Gather all experiment files (this will now include kurtosis_data folders)
    base_folder = "./experiment/visualization"
    print("Gathering experiment files...")
    organized_files = gather_experiment_files(base_folder, metric='kurtosis')
    
    # Output directory
    output_dir = "./experiment/visualization/plots/Kperp_across_diameter"
    
    # Get all available K values
    k_values = list(organized_files.keys())
    compartments = ['intra', 'extra']
    
    print(f"Found K values: {k_values}")
    
    # Generate plots for each K value and compartment
    for k_value in k_values:
        for compartment in compartments:
            print(f"\nProcessing {k_value} {compartment}...")
            plot_Kxy_across_diameters_for_K(
                organized_files, k_value, compartment, output_dir
            )
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()