import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from the core module
from simulation_toolkit.utils.metric_visualization.core import gather_experiment_files, extract_k_value_from_filename

def load_ADC_data_pickle(file_path):
    """Load pickle file and return ADC data."""
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def normalize_dz(dz_values, reference_value=None):
    """
    Normalize Dz values. If reference_value is None, use the first value.
    """
    if reference_value is None:
        reference_value = dz_values[0] if len(dz_values) > 0 else 1.0
    
    return dz_values / reference_value

def average_datasets_by_kappa(file_paths):
    """
    Calculate mean and std across multiple runs for each K value.
    
    Args:
        file_paths: List of file paths for the same diameter and compartment
    
    Returns:
        Dictionary with K value as key and averaged data as values
    """
    
    # Group files by K value
    grouped_data = defaultdict(list)
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        k_value = extract_k_value_from_filename(filename)
        
        if k_value:
            try:
                data = load_ADC_data_pickle(file_path)
                grouped_data[f'K{k_value}'].append(data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    # Calculate averages for each K value
    averaged_results = {}
    
    for k_value, datasets in grouped_data.items():
        if not datasets:
            continue
            
        # Collect all time series for averaging
        all_dz = []
        all_time = []
        
        for data in datasets:
            Dx, Dy, Dz, diff_time = data[:,0], data[:,1], data[:,2], data[:,3]
            all_dz.append(Dz)  # Parallel diffusion coefficient
            all_time.append(diff_time)
        
        # Convert to numpy arrays
        all_dz = np.array(all_dz)
        all_time = np.array(all_time)
        
        # Calculate mean and standard deviation
        mean_dz = np.mean(all_dz, axis=0)
        std_dz = np.std(all_dz, axis=0, ddof=1)
        mean_time = np.mean(all_time, axis=0)
        
        averaged_results[k_value] = {
            'Dz_mean': mean_dz,
            'Dz_std': std_dz,
            'diff_time_mean': mean_time,
            'num_runs': len(datasets)
        }
    
    return averaged_results

def plot_Dz_across_kappa_for_diameter(organized_files, diameter, compartment, output_dir):
    """
    Plot normalized Dz across different K values for a specific diameter and compartment
    vs 1/sqrt(t) with shaded error regions.
    
    Args:
        organized_files: Dictionary from gather_experiment_files()
        diameter: Diameter to plot (e.g., '1.68', '2.58')
        compartment: 'intra' or 'extra'
        output_dir: Directory to save plots
    """
    
    # Get all files for this diameter and compartment across all K values
    all_files = []
    
    for k_value, k_data in organized_files.items():
        if compartment in k_data and diameter in k_data[compartment]:
            all_files.extend(k_data[compartment][diameter])
    
    if not all_files:
        print(f"No files found for diameter {diameter} {compartment}")
        return
    
    # Average data by K value
    averaged_data = average_datasets_by_kappa(all_files)
    
    if not averaged_data:
        print(f"No valid data to plot for diameter {diameter} {compartment}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Colors for different K values (matching the reference file style)
    colors = {'K200': 'orange', 'K20': 'purple', 'K10': 'brown', 'K8': 'red'}
    
    # Plot each K value
    for k_value, data in sorted(averaged_data.items(), key=lambda x: int(x[0][1:]), reverse=True):
        diff_time_mean = data['diff_time_mean']
        dz_mean = data['Dz_mean']
        dz_std = data['Dz_std']
        
        # Calculate inverse square root of time
        inv_sqrt_time = 1.0 / np.sqrt(diff_time_mean)
        
        # Normalize Dz values (divide by first value to get D/D0)
        normalized_dz_mean = normalize_dz(dz_mean)
        
        # Calculate normalized standard deviation
        # For normalized data, we need to propagate the error
        reference_dz = dz_mean[0]
        normalized_dz_std = dz_std / reference_dz
        
        color = colors.get(k_value, 'black')
        kappa_num = k_value[1:]  # Remove 'K' prefix
        label = f'K={kappa_num}'
        
        # Plot with line and shaded error region
        ax.plot(inv_sqrt_time, normalized_dz_mean, color=color, label=label, linewidth=2)
        ax.fill_between(inv_sqrt_time, 
                       normalized_dz_mean - normalized_dz_std, 
                       normalized_dz_mean + normalized_dz_std,
                       color=color, alpha=0.3)
    
    # Formatting (matching the reference file style)
    ax.set_xlim([0.1, 1.0])
    ax.set_ylim([0.6, 1])
    ax.set_xlabel(r'$1/\sqrt{t}\;(\mathrm{ms}^{-1/2})$', fontsize=25)
    
    if compartment == 'intra':
        ax.set_ylabel(r'$D_{\mathrm{i},\!\!\parallel}/D_{0,\!\mathrm{i}}$', fontsize=25)
    elif compartment == 'extra': 
        ax.set_ylabel(r'$D_{\mathrm{e},\!\!\parallel}/D_{0,\!\mathrm{e}}$', fontsize=25)
    
    ax.set_title(r"$\overline{d}$="+str(diameter)+"µm", fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=18, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"Dparallel_normalized_vs_inv_sqrt_t_d{diameter}_{compartment}.png"
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
    output_dir = "./experiment/visualization/plots/Dparallel_normalized_across_kappa"
    
    # Get all available diameters (from any K value and compartment)
    all_diameters = set()
    compartments = ['intra', 'extra']
    
    for k_data in organized_files.values():
        for comp_data in k_data.values():
            all_diameters.update(comp_data.keys())
    
    all_diameters = sorted(list(all_diameters), key=float)
    
    print(f"Found diameters: {all_diameters}")
    
    # Generate plots for each diameter and compartment
    for diameter in all_diameters:
        for compartment in compartments:
            print(f"\nProcessing diameter {diameter} {compartment}...")
            plot_Dz_across_kappa_for_diameter(
                organized_files, diameter, compartment, output_dir
            )
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()