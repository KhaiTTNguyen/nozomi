from util import *
import matplotlib.cm as cm
import os 
import numpy as np
import pandas as pd
from collections import defaultdict

def import_array_ADC_full_path(file_name):
    print('file_name', file_name)
    Dx_Dy_Dz_difftime = load_ADC_data_pickle(file_name)
    return Dx_Dy_Dz_difftime

def load_ADC_data_pickle(file_name):
    # Open the Pickle file for reading in binary mode ('rb')
    with open(file_name, 'rb') as file:
        # Unpickle the data
        loaded_data = pickle.load(file)
    return loaded_data

def count_num_files_in_folder(folder_path):
    """Counts the number of files in a given directory (excluding subdirectories)."""
    count = 0
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            count += 1
    return count

def sort_lists_together(list_a, list_b):
    """Sorts list A and reorders list B based on the reordered positions of list A."""
    # Sort list A
    list_a_floats = [float(x) for x in list_a]
    # Create a dictionary mapping elements in list A to their original indices
    index_map = {element: index for index, element in enumerate(list_a_floats)}
    sorted_a_floats = sorted(list_a_floats, reverse=True)
    # Reorder list B based on the sorted order of list A
    reordered_b = [list_b[index_map[element]] for element in sorted_a_floats]
    return sorted_a_floats, reordered_b

def extract_diameter_from_file_path(file_path):
    # Split the path based on underscores (_)
    parts = file_path.split("_")
    # Find the index of the part starting with "avf_"
    avf_index = next((i for i, part in enumerate(parts) if part.startswith("avf")), None)
    # Check if "avf_" was found
    if avf_index is not None:
        pass
    else:
        print("'avf_' section not found in filepath")
    diameter_string = parts[avf_index+2]
    diameter = diameter_string[1:]
    return diameter

def get_path_to_first_file(folder_path):
    """Returns the full path to the first file found in a given folder."""
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return None

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            return file_path
    return None

def get_dispersion_value_kappa(file_path):
    import re
    match = re.findall(r'K(\d+)', file_path)

    if match:
        return match[0] # return the first occurence of a number after K
    else:
        print("No value found after 'K'")

def average_datasets_by_diameter(grouped_data):
    """
    Calculate both mean and standard deviation across multiple runs for each diameter.
    
    Args:
        grouped_data: Dictionary with diameter as keys and list of datasets as values
    
    Returns:
        Dictionary with averaged data and standard deviations
    """
    averaged_results = {}
    
    for diameter, multi_runs_data in grouped_data.items():
        if not multi_runs_data:
            continue
            
        # Initialize lists to store all time series for averaging
        all_dxy = []
        all_time = []
        
        # Collect all multi_runs_data for this diameter
        for data in multi_runs_data:
            Dx, Dy, Dz, diff_time = data[:,0], data[:,1], data[:,2], data[:,3]
            Dxy = (Dx + Dy) / 2
            all_dxy.append(Dxy)
            all_time.append(diff_time)
        
        # Convert to numpy arrays for easier manipulation
        all_dxy = np.array(all_dxy)
        all_time = np.array(all_time)
        
        # Calculate mean and standard deviation across runs
        mean_dxy = np.mean(all_dxy, axis=0)
        std_dxy = np.std(all_dxy, axis=0, ddof=1)  # Using sample standard deviation
        mean_time = np.mean(all_time, axis=0)
        
        # Store results including standard deviation
        averaged_results[diameter] = {
            'Dxy_mean': mean_dxy,
            'Dxy_std': std_dxy,
            'diff_time_mean': mean_time,
            'num_runs': len(multi_runs_data)
        }
    
    return averaged_results

def average_Kz_datasets_by_diameter(grouped_data):
    """
    Calculate both mean and standard deviation across multiple runs for each diameter.
    
    Args:
        grouped_data: Dictionary with diameter as keys and list of datasets as values
    
    Returns:
        Dictionary with averaged data and standard deviations
    """
    averaged_results = {}
    
    for diameter, multi_runs_data in grouped_data.items():
        if not multi_runs_data:
            continue
            
        # Initialize lists to store all time series for averaging
        all_kz = []
        all_time = []
        
        # Collect all multi_runs_data for this diameter
        for data in multi_runs_data:
            Dx, Dy, Dz, diff_time = data[:,0], data[:,1], data[:,2], data[:,3]
            all_kz.append(Dz)
            all_time.append(diff_time)
        
        # Convert to numpy arrays for easier manipulation
        all_kz = np.array(all_kz)
        all_time = np.array(all_time)
        
        # Calculate mean and standard deviation across runs
        mean_kz = np.mean(all_kz, axis=0)
        std_kz = np.std(all_kz, axis=0, ddof=1)  # Using sample standard deviation
        mean_time = np.mean(all_time, axis=0)
        
        # Store results including standard deviation
        averaged_results[diameter] = {
            'Kz_mean': mean_kz,
            'Kz_std': std_kz,
            'diff_time_mean': mean_time,
            'num_runs': len(multi_runs_data)
        }
    
    return averaged_results

def plot_Kxy_across_diameter(folder_path, diff_time_limit):
    '''
    Plot Dxy across diameters with mean and shaded standard deviation regions
    '''
    compartment = str(os.path.basename(folder_path))
    folder_name = os.path.join(folder_path, 'ADC_wrt_diameter')
    fig, ax = plt.subplots(figsize=(10,5))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")  
    
    num_files = count_num_files_in_folder(folder_path)
    print("Number of files:", num_files)

    path_to_first_file = get_path_to_first_file(folder_path)
    kappa = get_dispersion_value_kappa(path_to_first_file)
    
    # Group data by diameter
    grouped_data = defaultdict(list)
    
    # Process all pickle files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            
            # Extract diameter from filename
            diameter = extract_diameter_from_file_path(file_path)
            
            if diameter is not None:
                # Load data
                try:
                    with open(file_path, 'rb') as f:
                        Dx_Dy_Dz_difftime = pickle.load(f)
                        grouped_data[diameter].append(Dx_Dy_Dz_difftime)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Could not extract diameter from: {filename}")
    
    if not grouped_data:
        print("No valid data files found.")
        return
    
    # Calculate averages and standard deviations
    print("Calculating averages and standard deviations...")
    averaged_data = average_datasets_by_diameter(grouped_data)
    
    # Colors for different diameters
    colors = {'1.0': 'blue', '1.68': 'red', '2.58': 'green', '3.5': 'purple', '4.5': 'brown'}
    
    diameter_list = []
    
    # Plot with shaded error regions
    for diameter, data in averaged_data.items():
        diff_time_mean = data['diff_time_mean']
        dxy_mean = data['Dxy_mean']
        dxy_std = data['Dxy_std']
        
        # Filter data based on time limit
        time_mask = diff_time_mean <= diff_time_limit
        diff_time_filtered = diff_time_mean[time_mask]
        dxy_mean_filtered = dxy_mean[time_mask]
        dxy_std_filtered = dxy_std[time_mask]
        
        diameter_list.append(diameter)
        color_i = colors[diameter]
        
        # Plot mean line
        ax.plot(diff_time_filtered, dxy_mean_filtered, color=color_i, 
                   label=str(diameter)+' µm', linewidth=2)
        # ax.plot(diff_time_filtered, dxy_mean_filtered, color=color_i, 
        #            label=str(diameter)+' µm',  marker='.', markersize=8, linestyle='None')
        
        # Add shaded area for standard deviation
        ax.fill_between(diff_time_filtered, 
                       dxy_mean_filtered - dxy_std_filtered, 
                       dxy_mean_filtered + dxy_std_filtered,
                       color=color_i, alpha=0.3)
    
    # Handle legend with unique labels
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    unique_diameter_list = []
    for handle, label, diameter in zip(handles, labels, diameter_list):
        if label not in unique_labels:
            unique_labels[label] = handle
            unique_diameter_list.append(diameter) 
    
    print('Unique labels count:', len(list(unique_labels.keys())))
    
    # Sort legend by diameter
    sorted_diameter_list, reordered_labels = sort_lists_together(unique_diameter_list, list(unique_labels.keys()))
    sorted_diameter_list, reordered_handles = sort_lists_together(unique_diameter_list, list(unique_labels.values()))
    ax.legend(handles=reordered_handles, labels=reordered_labels, fontsize=17)

    # Save plot
    plot_file_name = os.path.join(folder_name, 'RD_wrt_diameter_with_errorbars'+'_K='+str(kappa)+'_difftime_limit'+str(diff_time_limit)+'.png')
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xlabel(r'$t\;(\mathrm{ms})$', fontsize=15)
    if compartment =='intra':
        ax.set_ylabel(r'$K_{\mathrm{i},\!\!\perp}$', fontsize=15) 
        ax.set_xlim([0.002, 100])
        ax.set_ylim([0.0, 10])
    elif compartment =='extra':
        ax.set_ylabel(r'$K_{\mathrm{e},\!\!\perp}$', fontsize=15) 
        ax.set_xlim([0.002, 20])
        ax.set_ylim([0.0, 0.8])
    
    ax.set_title(" $\kappa$="+str(kappa), fontsize=17)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(plot_file_name, dpi=500, bbox_inches='tight')
    print('Done plotting')

# def plot_Kz_across_diameter(folder_path, diff_time_limit):
#     '''
#     Plot Kz across diameters with mean and shaded standard deviation regions
#     '''
#     compartment = str(os.path.basename(folder_path))
#     folder_name = os.path.join(folder_path, 'ADC_wrt_diameter')
#     fig, ax = plt.subplots(figsize=(10,5))
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#         print("Folder created successfully.")
#     else:
#         print("Folder already exists.")  
    
#     num_files = count_num_files_in_folder(folder_path)
#     print("Number of files:", num_files)

#     path_to_first_file = get_path_to_first_file(folder_path)
#     kappa = get_dispersion_value_kappa(path_to_first_file)
    
#     # Group data by diameter
#     grouped_data = defaultdict(list)
    
#     # Process all pickle files in the directory
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.pkl'):
#             file_path = os.path.join(folder_path, filename)
            
#             # Extract diameter from filename
#             diameter = extract_diameter_from_file_path(file_path)
            
#             if diameter is not None:
#                 # Load data
#                 try:
#                     with open(file_path, 'rb') as f:
#                         Dx_Dy_Dz_difftime = pickle.load(f)
#                         grouped_data[diameter].append(Dx_Dy_Dz_difftime)
#                 except Exception as e:
#                     print(f"Error loading {filename}: {e}")
#             else:
#                 print(f"Could not extract diameter from: {filename}")
    
#     if not grouped_data:
#         print("No valid data files found.")
#         return
    
#     # Calculate averages and standard deviations
#     print("Calculating averages and standard deviations...")
#     averaged_data = average_Kz_datasets_by_diameter(grouped_data)
    
#     # Colors for different diameters
#     colors = {'1.0': 'blue', '1.68': 'red', '2.58': 'green', '3.5': 'purple', '4.5': 'brown'}
    
#     diameter_list = []
    
#     # Plot with shaded error regions
#     for diameter, data in averaged_data.items():
#         diff_time_mean = data['diff_time_mean']
#         kz_mean = data['Kz_mean']
#         kz_std = data['Kz_std']
        
#         # Filter data based on time limit
#         time_mask = diff_time_mean <= diff_time_limit
#         diff_time_filtered = diff_time_mean[time_mask]
#         kz_mean_filtered = kz_mean[time_mask]
#         kz_std_filtered = kz_std[time_mask]
        
#         diameter_list.append(diameter)
#         color_i = colors[diameter]
        
#         # Plot mean line
#         ax.plot(diff_time_filtered, kz_mean_filtered, color=color_i, 
#                    label=str(diameter)+' µm', linewidth=2)
        
#         # Add shaded area for standard deviation
#         ax.fill_between(diff_time_filtered, 
#                        kz_mean_filtered - kz_std_filtered, 
#                        kz_mean_filtered + kz_std_filtered,
#                        color=color_i, alpha=0.3)
    
#     # Handle legend with unique labels
#     handles, labels = ax.get_legend_handles_labels()
#     unique_labels = {}
#     unique_diameter_list = []
#     for handle, label, diameter in zip(handles, labels, diameter_list):
#         if label not in unique_labels:
#             unique_labels[label] = handle
#             unique_diameter_list.append(diameter) 
    
#     print('Unique labels count:', len(list(unique_labels.keys())))
    
#     # Sort legend by diameter
#     sorted_diameter_list, reordered_labels = sort_lists_together(unique_diameter_list, list(unique_labels.keys()))
#     sorted_diameter_list, reordered_handles = sort_lists_together(unique_diameter_list, list(unique_labels.values()))
#     # ax.legend(handles=reordered_handles, labels=reordered_labels, fontsize=17)

#     # Save plot
#     plot_file_name = os.path.join(folder_name, 'Kz_wrt_diameter_with_errorbars'+'_K='+str(kappa)+'_difftime_limit'+str(diff_time_limit)+'.png')
#     ax.tick_params(axis='both', which='major', labelsize=13)
#     ax.set_xlabel(r'$t\;(\mathrm{ms})$', fontsize=15)
#     if compartment =='intra':
#         ax.set_ylabel(r'$K_{\mathrm{i},\!\!\parallel}$', fontsize=15) 
#         ax.set_xlim([0.002, 100])
#         ax.set_ylim([0.0, 0.3])
#     elif compartment =='extra':
#         ax.set_ylabel(r'$K_{\mathrm{e},\!\!\parallel}$', fontsize=15) 
#         ax.set_xlim([0.002, 20])
#         ax.set_ylim([0.0, 0.8])
    
#     ax.set_title(" $\kappa$="+str(kappa), fontsize=17)
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
#     plt.savefig(plot_file_name, dpi=500, bbox_inches='tight')
#     print('Done plotting')


if __name__ == '__main__':
    RD_diff_time_limit = 100 #ms
    AD_diff_time_limit = 100
    
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_kurtosis_across_diameter/2025-10-03-K200/intra'
    plot_Kxy_across_diameter(folder_path, RD_diff_time_limit)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_kurtosis_across_diameter/2025-10-03-K200/extra'
    plot_Kxy_across_diameter(folder_path, RD_diff_time_limit)          
    
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_kurtosis_across_diameter/2025-10-03-K20/intra'
    plot_Kxy_across_diameter(folder_path, RD_diff_time_limit)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_kurtosis_across_diameter/2025-10-03-K20/extra'
    plot_Kxy_across_diameter(folder_path, RD_diff_time_limit)          
    
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_kurtosis_across_diameter/2025-10-03-K10/intra'
    plot_Kxy_across_diameter(folder_path, RD_diff_time_limit)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_kurtosis_across_diameter/2025-10-03-K10/extra'
    plot_Kxy_across_diameter(folder_path, RD_diff_time_limit)

    #  ---- Kz plots ----
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_kurtosis_across_diameter/2025-10-25-K200/intra'
    # plot_Kz_across_diameter(folder_path, RD_diff_time_limit)          
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_kurtosis_across_diameter/2025-10-25-K200/extra'
    # plot_Kz_across_diameter(folder_path, RD_diff_time_limit)          
    
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_kurtosis_across_diameter/2025-10-25-K20/intra'
    # plot_Kz_across_diameter(folder_path, RD_diff_time_limit)          
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_kurtosis_across_diameter/2025-10-25-K20/extra'
    # plot_Kz_across_diameter(folder_path, RD_diff_time_limit)          
    
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_kurtosis_across_diameter/2025-10-25-K10/intra'
    # plot_Kz_across_diameter(folder_path, RD_diff_time_limit)          
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_kurtosis_across_diameter/2025-10-25-K10/extra'
    # plot_Kz_across_diameter(folder_path, RD_diff_time_limit)