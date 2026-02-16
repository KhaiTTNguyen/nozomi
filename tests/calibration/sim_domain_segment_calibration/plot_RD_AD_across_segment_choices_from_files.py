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
import re
import glob

def count_num_files_in_folder(folder_path):
    """Counts the number of files in a given directory (excluding subdirectories)."""
    count = 0
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            count += 1
    return count

def extract_segment_number_from_filename(filename):
    """
    Extract segment number from filename.
    
    Example: diffcoeff_intra_5segments_...pkl -> returns 5
    """
    # Extract segment number from pattern like "5segments"
    segment_match = re.search(r'(\d+)segments', filename)
    segment_num = int(segment_match.group(1)) if segment_match else None
    return segment_num

def extract_compartment_from_filename(filename):
    """
    Extract compartment type (intra/extra) from filename.
    
    Example: diffcoeff_intra_5segments_...pkl -> returns "intra"
    """
    if 'intra' in filename:
        return 'intra'
    elif 'extra' in filename:
        return 'extra'
    else:
        return None

def average_datasets_by_segment(grouped_data):
    """
    Calculate both mean and standard deviation across multiple runs for each segment number.
    
    Args:
        grouped_data: Dictionary with segment number as keys and list of datasets as values
    
    Returns:
        Dictionary with averaged data and standard deviations
    """
    averaged_results = {}
    for segment_num, multi_runs_data in grouped_data.items():
        print(f'Segment {segment_num}: {len(multi_runs_data)} files')
        if not multi_runs_data:
            continue
            
        # Initialize lists to store all time series for averaging
        all_dxy = []
        all_dz = []
        all_time = []
        
        # Collect all multi_runs_data for this segment number
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
        averaged_results[segment_num] = {
            'Dxy_mean': mean_dxy,
            'Dxy_std': std_dxy,
            'Dz_mean': mean_dz,
            'Dz_std': std_dz,
            'diff_time_mean': mean_time,
            'num_runs': len(multi_runs_data)
        }
    
    return averaged_results

def plot_Dxy_across_segments(folder_path, diff_time_limit, compartment_filter=None):
    '''
    Plot Dxy (radial diffusivity) across different segment numbers with mean and shaded standard deviation regions
    '''
    folder_name = os.path.join(folder_path, 'figs')
    fig, ax = plt.subplots(figsize=(6,5))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")  
    
    num_files = count_num_files_in_folder(folder_path)
    print("Number of files:", num_files)

    # Group data by segment number
    grouped_data = defaultdict(list)
    
    # Process all pickle files in the directory
    pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
    
    for file_path in pkl_files:
        filename = os.path.basename(file_path)
        
        # Extract segment number and compartment type from filename
        segment_num = extract_segment_number_from_filename(filename)
        compartment = extract_compartment_from_filename(filename)
        
        # Filter by compartment if specified
        if compartment_filter and compartment != compartment_filter:
            continue
            
        if segment_num is not None and compartment is not None:
            # Load data
            try:
                with open(file_path, 'rb') as f:
                    Dx_Dy_Dz_difftime = pickle.load(f)
                    grouped_data[segment_num].append(Dx_Dy_Dz_difftime)
                    print(f"Loaded {filename}: {segment_num} segments, {compartment} compartment")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Could not extract segment number or compartment from: {filename}")
    
    if not grouped_data:
        print("No valid data files found.")
        return
    
    # Calculate averages and standard deviations
    print("Calculating averages and standard deviations...")
    averaged_data = average_datasets_by_segment(grouped_data)
    
    # Generate colors for different segment numbers
    segment_nums = sorted(averaged_data.keys())
    colors = cm.viridis(np.linspace(0, 1, len(segment_nums)))
    color_map = dict(zip(segment_nums, colors))
    
    # Plot with shaded error regions
    for segment_num, data in averaged_data.items():
        diff_time_mean = data['diff_time_mean']
        dxy_mean = data['Dxy_mean']
        dxy_std = data['Dxy_std']
        
        # Filter data based on time limit
        time_mask = diff_time_mean <= diff_time_limit
        diff_time_filtered = diff_time_mean[time_mask]
        dxy_mean_filtered = dxy_mean[time_mask]
        dxy_std_filtered = dxy_std[time_mask]
        
        print(f'Segment {segment_num}: mean std Dxy = {np.mean(dxy_std_filtered):.4f}')
        
        color = color_map[segment_num]
        label = f'{segment_num} segments'
        
        # Plot mean line
        ax.semilogx(diff_time_filtered, dxy_mean_filtered, color=color, 
                   label=label, linewidth=2)
        
        # Add shaded area for standard deviation
        ax.fill_between(diff_time_filtered, 
                       dxy_mean_filtered - dxy_std_filtered, 
                       dxy_mean_filtered + dxy_std_filtered,
                       color=color, alpha=0.3)
    
    ax.legend(fontsize=14)
    
    # Save plot
    compartment_suffix = f"_{compartment_filter}" if compartment_filter else ""
    plot_file_name = os.path.join(folder_name, f'RD_across_segments{compartment_suffix}_difftime_limit{diff_time_limit}.png')
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel(r'$t\;(\mathrm{ms})$', fontsize=25)
    ax.set_ylabel(r'$D_{\perp}\;(\mathrm{\mu m}^2/\mathrm{ms})$', fontsize=25) 
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.002, diff_time_limit])
    # ax.set_ylim([0.0, 2.7])  # Comment out to auto-scale
    fig.tight_layout()
    plt.savefig(plot_file_name, dpi=500, bbox_inches='tight')
    print(f'RD plot saved to: {plot_file_name}')
    plt.show()

def plot_Dz_across_segments(folder_path, diff_time_limit, compartment_filter=None):
    '''
    Plot Dz (axial diffusivity) across different segment numbers with mean and shaded standard deviation regions
    '''
    folder_name = os.path.join(folder_path, 'figs')
    fig, ax = plt.subplots(figsize=(6,5))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")  
    
    num_files = count_num_files_in_folder(folder_path)
    print("Number of files:", num_files)
 
    # Group data by segment number
    grouped_data = defaultdict(list)
    
    # Process all pickle files in the directory
    pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
    
    for file_path in pkl_files:
        filename = os.path.basename(file_path)
        
        # Extract segment number and compartment type from filename
        segment_num = extract_segment_number_from_filename(filename)
        compartment = extract_compartment_from_filename(filename)
        
        # Filter by compartment if specified
        if compartment_filter and compartment != compartment_filter:
            continue
            
        if segment_num is not None and compartment is not None:
            # Load data
            try:
                with open(file_path, 'rb') as f:
                    Dx_Dy_Dz_difftime = pickle.load(f)
                    grouped_data[segment_num].append(Dx_Dy_Dz_difftime)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Could not extract segment number or compartment from: {filename}")
    
    if not grouped_data:
        print("No valid data files found.")
        return
    
    # Calculate averages and standard deviations
    averaged_data = average_datasets_by_segment(grouped_data)
    
    # Generate colors for different segment numbers
    segment_nums = sorted(averaged_data.keys())
    colors = cm.viridis(np.linspace(0, 1, len(segment_nums)))
    color_map = dict(zip(segment_nums, colors))
    
    # Plot with shaded error regions
    for segment_num, data in averaged_data.items():
        diff_time = data['diff_time_mean']
        time_mask = diff_time <= diff_time_limit
        diff_time_filtered = diff_time[time_mask]
        inv_sqrt_time = 1.0 / np.sqrt(diff_time_filtered)
        dz_mean_filtered = data['Dz_mean'][time_mask]
        dz_std_filtered = data['Dz_std'][time_mask]
        
        print(f'Segment {segment_num}: mean std Dz = {np.mean(dz_std_filtered):.4f}')
        
        color = color_map[segment_num]
        label = f'{segment_num} segments'
        
        ax.plot(inv_sqrt_time, dz_mean_filtered, color=color, label=label, linewidth=2)
        ax.fill_between(inv_sqrt_time, 
                       dz_mean_filtered - dz_std_filtered, 
                       dz_mean_filtered + dz_std_filtered,
                       color=color, alpha=0.3)
    
    ax.legend(fontsize=14)
    
    # Save plot
    compartment_suffix = f"_{compartment_filter}" if compartment_filter else ""
    plot_file_name = os.path.join(folder_name, f'AD_across_segments{compartment_suffix}_difftime_limit{diff_time_limit}.png')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel(r'$1/\sqrt{t}\;(\mathrm{ms}^{-1/2})$', fontsize=25)
    ax.set_ylabel(r'$D_{\parallel}\;(\mathrm{\mu m}^2/\mathrm{ms})$', fontsize=25)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(plot_file_name, dpi=500, bbox_inches='tight')
    print(f'AD plot saved to: {plot_file_name}')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate RD and AD plots across different segment choices from pickle files')
    parser.add_argument('folder_path', help='Path to folder containing pickle files')
    parser.add_argument('--rd_time_limit', type=float, default=120.0, 
                       help='Time limit for RD plot in ms (default: 120)')
    parser.add_argument('--ad_time_limit', type=float, default=120.0,
                       help='Time limit for AD plot in ms (default: 120)')
    parser.add_argument('--compartment', choices=['intra', 'extra'], default=None,
                       help='Filter by compartment type (intra or extra). If not specified, includes all.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder_path):
        print(f"Error: Folder {args.folder_path} does not exist")
        return
    
    print(f"Scanning folder: {args.folder_path}")
    if args.compartment:
        print(f"Filtering by compartment: {args.compartment}")
    
    # Plot RD across different segment choices
    print("\nGenerating RD plot...")
    plot_Dxy_across_segments(args.folder_path, args.rd_time_limit, args.compartment)
    
    # Plot AD across different segment choices  
    print("\nGenerating AD plot...")
    plot_Dz_across_segments(args.folder_path, args.ad_time_limit, args.compartment)
    
    print("\nDone!")

if __name__ == "__main__":
    main()