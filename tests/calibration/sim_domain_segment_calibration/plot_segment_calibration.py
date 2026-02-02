import matplotlib.pyplot as plt
import numpy as np
# Data arrays
# 5: 17.96 / 13020.08
# 10: 64.36 / 4296.36
# 15: 194.52 / 2711.07
# 20: 449.37 / 2159.4
# 25: 865.61 / 2091.94
# 30: 1476.41 / 1797.39
# 35: 2383.17 / 1773.98

segments = [5, 10, 15, 20, 25, 30, 35]
# computation_times = [26883.31, 8411.21, 5534.94, 4442.01, 4407.09, 4439.89, 5235.64]
table_times =      np.array([17.96,     64.36,   194.52 ,  490.72, 865.61, 1476.41,  2383.17])
simulation_times = np.array([ 13020.08, 4296.36, 2711.07,  2159.4, 2091.94, 1797.39, 1773.98])
computation_times = simulation_times + table_times
# Create the plot
plt.figure(figsize=(6, 6))

# Create line without markers
plt.plot(segments, computation_times, 'b-', linewidth=2)
plt.plot(segments, table_times, 'g-', linewidth=2)
plt.plot(segments, simulation_times, 'r-', linewidth=2)

# Add red scatter points with legend
plt.scatter(segments, computation_times, color='b', s=50, zorder=5, label='Computation Time')
plt.scatter(segments, table_times, color='g', s=50, zorder=5, label='Table Time')
plt.scatter(segments, simulation_times, color='r', s=50, zorder=5, label='Simulation Time')

# Add labels and title
plt.xlabel('Number of Segments', fontsize=15)
plt.ylabel('Computation Time (sec)', fontsize=15)
plt.title('Computation Time based on Number of Segments', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Add grid for better readability
plt.grid(True, alpha=0.3)

# Add legend
plt.legend()

# Set y-axis to log scale
# plt.yscale('log')

# Improve layout
plt.tight_layout()

plt.figure(figsize=(6, 6))

# Create stacked area chart
plt.fill_between(segments, 0, table_times, color='lightgreen', alpha=0.7, label='Tabulation Time')
plt.fill_between(segments, table_times, computation_times, color='lightblue', alpha=0.7, label='Simulation Time')

# Add boundary lines for clarity
plt.plot(segments, computation_times, 'b-', linewidth=2, label='Total Computation Time')
plt.plot(segments, table_times, 'green', linewidth=1.5)
plt.scatter(segments, computation_times, color='b', s=50, zorder=5)
plt.scatter(segments, table_times, color='green', s=50, zorder=5)
# plt.scatter(segments, simulation_times, color='r', s=50, zorder=5, label='Simulation Time')

# Set y-axis to log scale
# plt.yscale('log')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Number of Segments', fontsize=15)
plt.ylabel('Run Time (sec)', fontsize=15)
plt.title('Run Time vs Number of Segments', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()