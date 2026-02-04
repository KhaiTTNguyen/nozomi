import pickle
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import pandas as pd

def plot_Dz_across_kappa(data_directory):
    """
    Main function to process pickle files and create plots with std regions.
    """
    compartment = str(os.path.basename(data_directory))
    
    # Directory containing pickle files
    output_directory = os.path.join(data_directory, 'ADC_wrt_kappa')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")
    # Group data by parameters
    grouped_data = defaultdict(list)
    
    # Process all pickle files in the directory
    for filename in os.listdir(data_directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(data_directory, filename)
            
            # Extract parameters from filename
            kappa, diameter = extract_parameters_from_filename(filename)
            
            if kappa is not None and diameter is not None:
                # Load data
                try:
                    Dx_Dy_Dz_difftime = load_pickle_data(filepath)
                    Dx_Dy_Dz_difftime = pd.DataFrame(Dx_Dy_Dz_difftime, columns=['Dx', 'Dy', 'Dz', 'diff_time'])
                    grouped_data[(kappa, diameter)].append(Dx_Dy_Dz_difftime)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Could not extract parameters from: {filename}")
    
    if not grouped_data:
        print("No valid data files found. Please check the data directory and filename patterns.")
        return
    
    # Calculate averages and standard deviations
    print("\nCalculating averages and standard deviations...")
    averaged_data = average_datasets(grouped_data)
    
    # Create plots with error bars
    print("\nGenerating plots with error bars...")
    plot_diffusion_analysis(averaged_data, compartment, output_directory)
    
    print(f"Plots saved to: {output_directory}")

def extract_parameters_from_filename(filename):
    """
    Extract kappa and diameter values from filename.
    Assumes filename format contains these parameters.
    """
    # Pattern to extract kappa and diameter from filename
    # Adjust this regex pattern based on your actual filename format
    kappa = re.findall(r'K(\d+)', filename)

    if kappa:
        kappa = kappa[0] # return the first occurence of a number after K
    else:
        print("No value found after 'K'")
    
    parts = filename.split("_")
    # Find the index of the part starting with "avf_"
    avf_index = next((i for i, part in enumerate(parts) if part.startswith("avf")), None)
    # Check if "avf_" was found
    if avf_index is not None:
        pass
    else:
        print("'avf_' section not found in filepath")
    diameter_string=parts[avf_index+2]
    diameter = diameter_string[1:]
    
    return kappa, diameter

def load_pickle_data(filepath):
    """
    Load data from pickle file.
    Expected to contain Dx, Dy, Dz, diff_time arrays.
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def average_datasets(grouped_data):
    """
    Calculate both mean and standard deviation
    across multiple runs for each parameter combination.
    
    Args:
        grouped_data: Dictionary with (kappa, diameter) as keys and list of datasets as values
    
    Returns:
        Dictionary with averaged data and standard deviations
    """
    averaged_results = {}
    
    for (kappa, diameter), multi_runs_data in grouped_data.items():
        if not multi_runs_data:
            continue
            
        # Initialize lists to store all time series for averaging
        all_dz = []
        all_time = []
        
        # Collect all multi_runs_data for this parameter combination
        for data in multi_runs_data:
            all_dz.append(data['Dz'])
            all_time.append(data['diff_time'])
        
        # Convert to numpy arrays for easier manipulation
        all_dz = np.array(all_dz)
        all_time = np.array(all_time)
        # Calculate mean and standard deviation across runs
        mean_dz = np.mean(all_dz, axis=0)
        std_dz = np.std(all_dz, axis=0, ddof=1)  # Using sample standard deviation
        mean_time = np.mean(all_time, axis=0)
        
        # Store results including standard deviation
        averaged_results[(kappa, diameter)] = {
            'Dz_mean': mean_dz,
            'Dz_std': std_dz,
            'diff_time_mean': mean_time,
            'Dx_mean': np.mean([data['Dx'] for data in multi_runs_data], axis=0),
            'Dy_mean': np.mean([data['Dy'] for data in multi_runs_data], axis=0),
            'num_runs': len(multi_runs_data)
        }
    
    return averaged_results

def normalize_dz(dz_values, reference_value=None):
    """
    Normalize Dz values. If reference_value is None, use the first value.
    """
    if reference_value is None:
        reference_value = dz_values[0] if len(dz_values) > 0 else 1.0
    
    return dz_values / reference_value

def sort_lists_together(list_a, list_b):
    """Sorts list A and reorders list B based on the reordered positions of list A.

    Args:
    list_a: The list to be sorted.
    list_b: The list to be reordered.

    Returns:
    A tuple containing the sorted list A and the reordered list B.
    """
    # Sort list A
    list_a_floats = [float(x) for x in list_a]
    # Create a dictionary mapping elements in list A to their original indices
    index_map = {element: index for index, element in enumerate(list_a_floats)}
    sorted_a_floats = sorted(list_a_floats, reverse=True)
    # Reorder list B based on the sorted order of list A
    reordered_b = [list_b[index_map[element]] for element in sorted_a_floats]

    return sorted_a_floats, reordered_b

def plot_diffusion_analysis(averaged_data, compartment, output_dir=None):
    """
    Plot normalized Dz vs inverse square root of time with error bars
    and reduced marker size.
    """
    fig, ax = plt.subplots(figsize=(5,5))

    # Color map for different parameter combinations
    colors = {'200': 'orange', '20': 'purple', '10': 'brown'}
    # colors = [ 'orange', 'purple', 'brown', 'c', 'm', 'y', 'k']
    
    color_idx = 0
    kappa_list=[]
    for (kappa, diameter), mean_data in averaged_data.items():
        # Calculate inverse square root of time
        inv_sqrt_time = 1.0 / np.sqrt(mean_data['diff_time_mean'])
        
        # Normalize Dz values
        normalized_dz_mean = normalize_dz(mean_data['Dz_mean'])
        
        
        # Calculate normalized standard deviation
        # For normalized data, we need to propagate the error
        reference_dz = mean_data['Dz_mean'][0]
        normalized_dz_std = mean_data['Dz_std'] / reference_dz
        # Create label for legend
        label='K='+str(kappa)
        color=colors[kappa]
        # Plot with error bars and reduced marker size
        # Add shaded area for standard deviation
        ax.plot(inv_sqrt_time, normalized_dz_mean, color=color, label=label, linewidth=2)
        ax.fill_between(inv_sqrt_time, 
                       normalized_dz_mean - normalized_dz_std, 
                       normalized_dz_mean + normalized_dz_std,
                       color=color, alpha=0.3)
        kappa_list.append(kappa)
        color_idx += 1
    ax.set_xlim([0.1, 1.0])
    ax.set_ylim([0.6, 1])
    ax.set_xlabel(r'$1/\sqrt{t}\;(\mathrm{ms}^{-1/2})$', fontsize=25)
    if compartment =='intra':
        ax.set_ylabel(r'$D_{\mathrm{i},\!\!\parallel}/D_{0,\!\mathrm{i}}$', fontsize=25)
     
    elif compartment =='extra': 
        ax.set_ylabel(r'$D_{\mathrm{e},\!\!\parallel}/D_{0,\!\mathrm{e}}$', fontsize=25)

    
    ax.set_title(r"$\overline{d}$="+str(diameter)+"µm", fontsize=25)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    unqiue_kappa_list = []
    for handle, label, kappa in zip(handles, labels, kappa_list):
        if label not in unique_labels:
            unique_labels[label] = handle
            unqiue_kappa_list.append(kappa) 
    print('list(unique_labels.keys())', len((list(unique_labels.keys()))))
    print('list(unique_labels.values())', len(list(unique_labels.values())))       
    sorted_kappa_list, reordered_labels = sort_lists_together(unqiue_kappa_list, list(unique_labels.keys()))
    sorted_kappa_list, reordered_handles = sort_lists_together(unqiue_kappa_list, list(unique_labels.values()))
    ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.legend(handles=reordered_handles, labels=reordered_labels, fontsize=18, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    # Save plot if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'AD_with_errorbars.png'), 
                   dpi=500, bbox_inches='tight')
        print('Done plotting')

# # Additional utility function for custom error bar styling
# def plot_diffusion_analysis_custom_style(averaged_data, output_dir=None):
#     """
#     Alternative plotting function with custom error bar styling.
#     """
#     plt.figure(figsize=(14, 10))
    
#     # Custom color palette
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
#               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
#     markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
#     for idx, ((kappa, diameter), data) in enumerate(averaged_data.items()):
#         inv_sqrt_time = 1.0 / np.sqrt(data['diff_time_mean'])
#         normalized_dz_mean = normalize_dz(data['Dz_mean'])
        
#         # Calculate normalized standard deviation
#         reference_dz = data['Dz_mean'][0]
#         normalized_dz_std = data['Dz_std'] / reference_dz
        
#         label = f'U+3ba={kappa}, d={diameter} (n={data["num_runs"]})'
        
#         # Plot with custom styling
#         plt.errorbar(
#             inv_sqrt_time, 
#             normalized_dz_mean, 
#             yerr=normalized_dz_std,
#             marker=markers[idx % len(markers)], 
#             markersize=5,  # Slightly larger than the basic version but still reduced
#             color=colors[idx % len(colors)],
#             label=label,
#             capsize=4,
#             capthick=1.5,
#             elinewidth=1.2,
#             linestyle='-',
#             linewidth=2,
#             alpha=0.85,
#             markeredgecolor='white',
#             markeredgewidth=0.5
#         )
    
#     plt.xlabel('1/√(time) [1/√s]', fontsize=14, fontweight='bold')
#     plt.ylabel('Normalized Dz', fontsize=14, fontweight='bold')
#     plt.title('Diffusion Analysis: Normalized Dz vs 1/√(time)\nwith Standard Deviation Error Bars', 
#               fontsize=16, fontweight='bold', pad=20)
    
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
#     plt.grid(True, alpha=0.4, linestyle='--')
#     plt.tight_layout()
    
#     # Improve axis formatting
#     plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         plt.savefig(os.path.join(output_dir, 'diffusion_analysis_custom_style.png'), 
#                    dpi=300, bbox_inches='tight')
#         plt.savefig(os.path.join(output_dir, 'diffusion_analysis_custom_style.pdf'), 
#                    bbox_inches='tight')
    
#     plt.show()

if __name__ == "__main__":
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d168/extra'
    plot_Dz_across_kappa(folder_path)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d168/intra'
    plot_Dz_across_kappa(folder_path)        

    # d2.58
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d258/extra'
    plot_Dz_across_kappa(folder_path)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d258/intra'
    plot_Dz_across_kappa(folder_path)   
    
    # d3.5
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d35/extra'
    plot_Dz_across_kappa(folder_path)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d35/intra'
    plot_Dz_across_kappa(folder_path)          

    # d4.5
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d45/extra'
    plot_Dz_across_kappa(folder_path)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d45/intra'
    plot_Dz_across_kappa(folder_path)          
    