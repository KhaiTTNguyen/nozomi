import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import re
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import simulation_toolkit.toolkit_params as config_params

def parse_filename(filename):
    """
    Extract segment number, table time, and simulation time from filename.
    
    Example filename:
    diffcoeff_intra_5segments_500000spins_array500_fibers_boxL_68.0_2026-02-13_17-37_avf_0.67_d1.68_sig0.45wo10_wc3_wl3_K200_ODI_0.0032_11977.67_sec_TABLEtime16.1sec_SIMtime125.76sec_timestep0.002ms.pkl
    """
    # Extract segment number from pattern like "5segments"
    segment_match = re.search(r'(\d+)segments', filename)
    segment_num = int(segment_match.group(1)) if segment_match else None
    
    # Extract table time from pattern like "TABLEtime16.1sec"
    table_time_match = re.search(r'TABLEtime([\d.]+)sec', filename)
    table_time = float(table_time_match.group(1)) if table_time_match else None
    
    # Extract simulation time from pattern like "SIMtime125.76sec"
    sim_time_match = re.search(r'SIMtime([\d.]+)sec', filename)
    sim_time = float(sim_time_match.group(1)) if sim_time_match else None
    
    return segment_num, table_time, sim_time

def collect_data_from_files(folder_path):
    """
    Collect segment number, table times, and simulation times from all .pkl files in folder.
    """
    # Find all .pkl files in the folder
    pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
    
    if not pkl_files:
        print(f"No .pkl files found in {folder_path}")
        return None, None, None
    
    # Dictionary to store data by segment number
    data_dict = defaultdict(list)
    
    for pkl_file in pkl_files:
        filename = os.path.basename(pkl_file)
        segment_num, table_time, sim_time = parse_filename(filename)
        
        if segment_num is not None and table_time is not None and sim_time is not None:
            data_dict[segment_num].append((table_time, sim_time))
            print(f"  Segments: {segment_num}, Table time: {table_time}s, Sim time: {sim_time}s")
        else:
            print(f"Warning: Could not parse all values from {filename}")
    
    if not data_dict:
        print("No valid data extracted from filenames")
        return None, None, None
    
    # Sort segments and average times if multiple files per segment
    segments = sorted(data_dict.keys())
    table_times = []
    simulation_times = []
    
    for segment in segments:
        # Average the times if multiple files for same segment count
        times_data = data_dict[segment]
        avg_table_time = np.mean([t[0] for t in times_data])
        avg_sim_time = np.mean([t[1] for t in times_data])
        
        table_times.append(avg_table_time)
        simulation_times.append(avg_sim_time)
        
        if len(times_data) > 1:
            print(f"Segment {segment}: Averaged {len(times_data)} files - Table: {avg_table_time:.2f}s, Sim: {avg_sim_time:.2f}s")
    
    return np.array(segments), np.array(table_times), np.array(simulation_times)

def create_plot(segments, table_times, simulation_times, output_folder=None):
    """
    Create the segment calibration plot similar to the original.
    """
    computation_times = simulation_times + table_times
    
    # Styling constants
    FIGURE_SIZE = (6, 6)
    TITLE_FONTSIZE = 16
    LABEL_FONTSIZE = 15
    TICK_FONTSIZE = 14
    LINE_WIDTH = 2
    SCATTER_SIZE = 50
    
    # Create the plot
    plt.figure(figsize=FIGURE_SIZE)
    
    # Create stacked area chart
    plt.fill_between(segments, 0, table_times, color='lightgreen', alpha=0.7, label='Tabulation Time')
    plt.fill_between(segments, table_times, computation_times, color='lightblue', alpha=0.7, label='Simulation Time')
    
    # Add boundary lines and scatter points
    plt.plot(segments, computation_times, 'b-', linewidth=LINE_WIDTH, label='Total Computation Time')
    plt.plot(segments, table_times, 'green', linewidth=1.5)
    plt.scatter(segments, computation_times, color='b', s=SCATTER_SIZE, zorder=5)
    plt.scatter(segments, table_times, color='green', s=SCATTER_SIZE, zorder=5)
    
    # Formatting
    plt.xlabel('Number of Segments', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Run Time (sec)', fontsize=LABEL_FONTSIZE)
    plt.title('Run Time vs Number of Segments', fontsize=TITLE_FONTSIZE, fontweight='bold')
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create output folder and save
    if output_folder is None:
        output_folder = "./tests/calibration/sim_domain_segment_calibration/calibrate_segments_vs_runtime/figs"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plot_filepath = os.path.join(output_folder, "segment_calibration_plot.png")
    plt.savefig(plot_filepath, dpi=500, bbox_inches='tight')
    print(f"\Plot saved to: {plot_filepath}")
    
    # Print summary
    print("\nSummary of extracted data:")
    print("Segments:", segments.tolist())
    print("Table times (s):", [f"{t:.2f}" for t in table_times])
    print("Simulation times (s):", [f"{t:.2f}" for t in simulation_times])
    print("Total computation times (s):", [f"{t:.2f}" for t in computation_times])

def main():
    parser = argparse.ArgumentParser(description='Generate segment calibration plot from pickle files')
    parser.add_argument('folder_path', help='Path to folder containing pickle files')
    parser.add_argument('-o', '--output', help='Output folder for plot (optional)', default=None)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder_path):
        print(f"Error: Folder {args.folder_path} does not exist")
        return

    # Collect data from files
    segments, table_times, simulation_times = collect_data_from_files(args.folder_path)
    
    if segments is None:
        print("Failed to extract data from files")
        return
    
    # Create plot
    create_plot(segments, table_times, simulation_times, args.output)

if __name__ == "__main__":
    main()