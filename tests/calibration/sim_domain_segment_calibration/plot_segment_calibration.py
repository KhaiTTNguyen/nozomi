import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import simulation_toolkit.toolkit_params as config_params

'''
This script generates a plot of run time vs number of segments for the segment calibration process.
The number of segments were varied between [5, 10, 15, 20, 25, 30, 35]

Tabulation times and Simulation times for were taken from Monte Carlo simulations 
of diffusion in a substrate with 2236 fibers, 
GEV-distributed \bar{d} of 1.75 μm, \kappa = 200, and V_{in} = 66%.
'''
segments = [5, 10, 15, 20, 25, 30, 35]
table_times = np.array([17.96, 64.36, 194.52, 490.72, 865.61, 1476.41, 2383.17])
simulation_times = np.array([13020.08, 4296.36, 2711.07, 2159.4, 2091.94, 1797.39, 1773.98])
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

# Create results directory and save
config_params.NUM_SEGMENT_CALIBRATION_FOLDER_PATH = "./tests/calibration/sim_domain_segment_calibration/calibrate_segments_vs_runtime/figs" 

if not os.path.exists(config_params.NUM_SEGMENT_CALIBRATION_FOLDER_PATH):
    os.makedirs(config_params.NUM_SEGMENT_CALIBRATION_FOLDER_PATH)

plot_filepath = os.path.join(config_params.NUM_SEGMENT_CALIBRATION_FOLDER_PATH, "segment_calibration_plot.png")
print(plot_filepath)
plt.savefig(plot_filepath, dpi=500, bbox_inches='tight')
plt.close()
    