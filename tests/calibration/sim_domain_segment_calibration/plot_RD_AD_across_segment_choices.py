import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os 
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib.ticker import FormatStrFormatter
import glob

def count_num_files_in_folder(folder_path):
    """Counts the number of files in a given directory (excluding subdirectories)."""
    count = 0
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            count += 1
    return count

def extract_compartment_name_from_filename(filename):
    """
    Extract compartment type from filename.
    
    Looks for 'intra', 'extra', or 'combined' in the filename.
    """
    filename_lower = filename.lower()
    
    if 'intra' in filename_lower:
        return 'intra'
    elif 'extra' in filename_lower:
        return 'extra'
    elif 'combined' in filename_lower:
        return 'combined'
    else:
        # Fallback: split by underscores and look for compartment names
        parts = filename.split("_")
        for part in parts:
            part_lower = part.lower()
            if part_lower in ['intra', 'extra', 'combined']:
                return part_lower
    
    print(f"Warning: 'intra', 'extra', or 'combined' not found in filename: {filename}")
    return None

def average_datasets_by_compartment(grouped_data):
    """
    Calculate both mean and standard deviation across multiple runs for each compartment.
    
    Args:
        grouped_data: Dictionary with compartment as keys and list of datasets as values
    
    Returns:
        Dictionary with averaged data and standard deviations
    """
    averaged_results = {}
    for compartment, multi_runs_data in grouped_data.items():
        print(f'{compartment}: {len(multi_runs_data)} files')
        if not multi_runs_data:
            continue
            
        # Initialize lists to store all time series for averaging
        all_dxy = []
        all_dz = []
        all_time = []
        
        # Collect all multi_runs_data for this compartment
        for data in multi_runs_data:
            Dx, Dy, Dz, diff_time = data[:,0], data[:,1], data[:,2], data[:,3]
            Dxy = (Dx + Dy) / 2
            all_dxy.append(Dxy)
            all_dz.append(Dz)
            all_time.append(diff_time)
        
        # Convert to numpy arrays for easier manipulation
        all_dxy = np.array(all_dxy)
        all_dz = np.array(all_dz)
        all_time = np.array(all_time)
        
        # Calculate mean and standard deviation across runs
        mean_dxy = np.mean(all_dxy, axis=0)
        std_dxy = np.std(all_dxy, axis=0, ddof=1)  # Using sample standard deviation
        mean_dz = np.mean(all_dz, axis=0)
        std_dz = np.std(all_dz, axis=0, ddof=1)
        mean_time = np.mean(all_time, axis=0)
        
        # Store results including standard deviation
        averaged_results[compartment] = {
            'Dxy_mean': mean_dxy,
            'Dxy_std': std_dxy,
            'Dz_mean': mean_dz,
            'Dz_std': std_dz,
            'diff_time_mean': mean_time,
            'num_runs': len(multi_runs_data)
        }
    
    return averaged_results

def plot_Dxy_across_compartment(folder_path, diff_time_limit, output_folder):
    '''
    Plot Dxy across compartments with mean and shaded standard deviation regions
    '''
    fig, ax = plt.subplots(figsize=(6,5))
    
    num_files = count_num_files_in_folder(folder_path)
    print("Number of files:", num_files)

    # Group data by compartment
    grouped_data = defaultdict(list)
    
    # Process all pickle files in the directory
    pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
    
    for file_path in pkl_files:
        filename = os.path.basename(file_path)
        
        # Extract compartment from filename
        compartment = extract_compartment_name_from_filename(filename)
        
        if compartment is not None:
            # Load data
            try:
                with open(file_path, 'rb') as f:
                    Dx_Dy_Dz_difftime = pickle.load(f)
                    grouped_data[compartment].append(Dx_Dy_Dz_difftime)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Could not extract compartment from: {filename}")
    
    if not grouped_data:
        print("No valid data files found.")
        return
    
    # Calculate averages and standard deviations
    print("Calculating averages and standard deviations...")
    averaged_data = average_datasets_by_compartment(grouped_data)
    
    # Colors for different compartments
    colors = {
        'intra': '#FF8C00',    # Dark orange
        'extra': '#008B8B',    # Dark cyan/teal
        'combined': '#800080'  # Purple
    }
    compartment_list = []
    
    # Plot with shaded error regions
    for compartment, data in averaged_data.items():
        diff_time_mean = data['diff_time_mean']
        dxy_mean = data['Dxy_mean']
        dxy_std = data['Dxy_std']
        
        # Filter data based on time limit
        time_mask = diff_time_mean <= diff_time_limit
        diff_time_filtered = diff_time_mean[time_mask]
        dxy_mean_filtered = dxy_mean[time_mask]
        dxy_std_filtered = dxy_std[time_mask]
        compartment_list.append(compartment)
        color_i = colors.get(compartment, '#000000')  # Default to black if color not found
        
        # Plot mean line
        ax.semilogx(diff_time_filtered, dxy_mean_filtered, color=color_i, 
                   label=f'{compartment}-axonal', linewidth=2)
        
        # Add shaded area for standard deviation
        ax.fill_between(diff_time_filtered, 
                       dxy_mean_filtered - dxy_std_filtered, 
                       dxy_mean_filtered + dxy_std_filtered,
                       color=color_i, alpha=0.3)
    
    # Handle legend with unique labels
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    unique_compartment_list = []
    for handle, label, compartment in zip(handles, labels, compartment_list):
        if label not in unique_labels:
            unique_labels[label] = handle
            unique_compartment_list.append(compartment) 
    
    ax.legend(handles=list(unique_labels.values()), labels=list(unique_labels.keys()), fontsize=18)

    # Save plot
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    plot_file_name = os.path.join(output_folder, f'RD_across_segment_choices_difftime_limit{diff_time_limit}ms.png')
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel(r'$t\;(\mathrm{ms})$', fontsize=25)
    ax.set_ylabel(r'$D_{\perp}\;(\mathrm{\mu m}^2/\mathrm{ms})$', fontsize=25) 
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.002, diff_time_limit])
    ax.set_ylim([0.0, 2.7])
    fig.tight_layout()
    plt.savefig(plot_file_name, dpi=500, bbox_inches='tight')
    plt.show()

def plot_Dz_across_compartment(folder_path, diff_time_limit, output_folder):
    '''
    Plot Dz across compartments with mean and shaded standard deviation regions
    '''
    fig, ax = plt.subplots(figsize=(6,5))
    
    num_files = count_num_files_in_folder(folder_path)
    print("Number of files:", num_files)
 
    # Group data by compartment
    grouped_data = defaultdict(list)
    
    # Process all pickle files in the directory
    pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
    
    for file_path in pkl_files:
        filename = os.path.basename(file_path)
        
        # Extract compartment from filename
        compartment = extract_compartment_name_from_filename(filename)
        
        if compartment is not None:
            # Load data
            try:
                with open(file_path, 'rb') as f:
                    Dx_Dy_Dz_difftime = pickle.load(f)
                    grouped_data[compartment].append(Dx_Dy_Dz_difftime)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Could not extract compartment from: {filename}")
    
    if not grouped_data:
        print("No valid data files found.")
        return
    
    # Calculate averages and standard deviations for Dz
    averaged_results = average_datasets_by_compartment(grouped_data)
    
    # Colors for different compartments
    colors = {
        'intra': '#FF8C00',    # Dark orange
        'extra': '#008B8B',    # Dark cyan/teal
        'combined': '#800080'  # Purple
    }
    compartment_list = []
    
    # Plot with shaded error regions
    for compartment, data in averaged_results.items():
        diff_time = data['diff_time_mean']
        time_mask = diff_time <= diff_time_limit
        diff_time_filtered = diff_time[time_mask]
        inv_sqrt_time = 1.0 / np.sqrt(diff_time_filtered)
        dz_mean_filtered = data['Dz_mean'][time_mask]
        dz_std_filtered = data['Dz_std'][time_mask]
        
        compartment_list.append(compartment)
        color = colors.get(compartment, '#000000')  # Default to black if color not found
        label = f'{compartment}-axonal'
        ax.plot(inv_sqrt_time, dz_mean_filtered, color=color, label=label, linewidth=2)
        ax.fill_between(inv_sqrt_time, 
                       dz_mean_filtered - dz_std_filtered, 
                       dz_mean_filtered + dz_std_filtered,
                       color=color, alpha=0.3)
    
    # Handle legend with unique labels
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    unique_compartment_list = []
    for handle, label, compartment in zip(handles, labels, compartment_list):
        if label not in unique_labels:
            unique_labels[label] = handle
            unique_compartment_list.append(compartment) 
    
    ax.legend(handles=list(unique_labels.values()), labels=list(unique_labels.keys()), fontsize=18)
    
    # Save plot
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    plot_file_name = os.path.join(output_folder, f'AD_across_segment_choices_difftime_limit{diff_time_limit}ms.png')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel(r'$1/\sqrt{t}\;(\mathrm{ms}^{-1/2})$', fontsize=25)
    ax.set_ylabel(r'$D_{\parallel}\;(\mathrm{\mu m}^2/\mathrm{ms})$', fontsize=25)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(plot_file_name, dpi=500, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate RD and AD plots grouped by compartment from ADC data files')
    parser.add_argument('input_folder', help='Path to ADCdata folder containing pickle files')
    # Default output: sibling `figs/` folder next to this script.
    _default_out = str(Path(__file__).resolve().parent /
                       'validate_same_results_with_same_substrate_different_segments_choices' /
                       'figs')
    parser.add_argument('-o', '--output',
                       default=_default_out,
                       help='Output folder for plots')
    parser.add_argument('--rd_time_limit', type=float, default=120.0, 
                       help='Time limit for RD plot in ms (default: 120)')
    parser.add_argument('--ad_time_limit', type=float, default=120.0,
                       help='Time limit for AD plot in ms (default: 120)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder {args.input_folder} does not exist")
        return
    
    # Plot RD across different compartments
    print("\nGenerating RD plot...")
    plot_Dxy_across_compartment(args.input_folder, args.rd_time_limit, args.output)
    
    # Plot AD across different compartments  
    print("\nGenerating AD plot...")
    plot_Dz_across_compartment(args.input_folder, args.ad_time_limit, args.output)
    
    print(f"Output folder: {args.output}")
    print("\nDone!")

if __name__ == "__main__":
    main()