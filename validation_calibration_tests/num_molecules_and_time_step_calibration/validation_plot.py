import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_plot_from_csv(csv_filepath):
    '''Load summary results from CSV and create analysis plots'''
    
    # Load the data
    df = pd.read_csv(csv_filepath)
    
    print(f"Loaded data with {len(df)} parameter combinations")
    print(f"Columns: {list(df.columns)}")
    print(f"Molecule counts: {sorted(df['molecules'].unique())}")
    print(f"Time steps: {sorted(df['time_step'].unique())}")
    
    # Extract parameters from filename or set defaults
    # You can modify these based on your actual values
    a = 0.5  # cylinder radius
    D0 = 2.0  # diffusion coefficient
    time_threshold = 0.03  # time threshold
    
    # Get the directory where the CSV is located to save plots there
    base_dir = os.path.dirname(csv_filepath)
    
    # Create the analysis plots
    create_analysis_plots_from_df(df, base_dir, a, D0, time_threshold, csv_filepath)
    
    return df

def create_analysis_plots_from_df(df, base_dir, a, D0, time_threshold, csv_filepath):
    '''Create comprehensive analysis plots from DataFrame'''
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: mae vs Molecules for different time steps
    time_steps = df['time_step'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_steps)))
    
    for i, ts in enumerate(sorted(time_steps)):
        data = df[df['time_step'] == ts].sort_values('molecules')
        ax1.loglog(data['molecules'], data['mae'], 'o-', 
                  color=colors[i], label=f'dt={ts}', linewidth=2, markersize=6)
        ax1.fill_between(data['molecules'], 
                        data['mae'] - data['std_mae'],
                        data['mae'] + data['std_mae'],
                        color=colors[i], alpha=0.2)
    
    ax1.set_xlabel('Number of Molecules', fontsize=17)
    ax1.set_ylabel(f'Mean mae (%)', fontsize=17)
    ax1.set_title(f'mae vs Number of Molecules\n', fontsize=17)
    ax1.legend(ncol=4, fontsize=10)
    ax1.grid(True, which="both", alpha=0.8)
    ax1.set_xlim(left=df['molecules'].min() * 0.8)
    ax1.set_ylim(bottom=df['mae'].min() * 0.8, top=(df['mae']+df['std_mae']).max() * 1.3)
    
    # Plot 2: mae vs Time Step for different molecules
    molecules_list = df['molecules'].unique()
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(molecules_list)))
    
    for i, mol in enumerate(sorted(molecules_list)):
        data = df[df['molecules'] == mol].sort_values('time_step')
        ax2.loglog(data['time_step'], data['mae'], 's-', 
                  color=colors2[i], label=f'{mol:.0e} molecules', linewidth=2, markersize=6)
        ax2.fill_between(data['time_step'], 
                        data['mae'] - data['std_mae'],
                        data['mae'] + data['std_mae'],
                        color=colors2[i], alpha=0.2)
    
    ax2.set_xlabel('Time Step (ms)', fontsize=17)
    ax2.set_ylabel(f'Mean mae (%)', fontsize=17)
    ax2.set_title(f'mae vs Time Step\n(t >= {time_threshold} ms)', fontsize=17)
    ax2.legend(ncol=2, fontsize=10)
    ax2.grid(True,which="both", alpha=0.8)
    ax2.set_xlim(left=df['time_step'].min() * 0.8)
    ax2.set_ylim(bottom=df['mae'].min() * 0.8)
    
    # Plot 3: Computation Time vs Molecules
    for i, ts in enumerate(sorted(time_steps)):
        data = df[df['time_step'] == ts].sort_values('molecules')
        ax3.loglog(data['molecules'], data['mean_computation_time'], 'o-', 
                  color=colors[i], label=f'dt={ts}', linewidth=2, markersize=6)
    
    ax3.set_xlabel('Number of Molecules', fontsize=17)
    ax3.set_ylabel('Mean Computation Time (s)', fontsize=17)
    ax3.set_title('Computation Time vs Number of Molecules', fontsize=17)
    ax3.legend(ncol=2, fontsize=10)
    ax3.grid(True, which="both", alpha=0.8)
    ax3.set_xlim(left=df['molecules'].min() * 0.8)
    ax3.set_ylim(bottom=df['mean_computation_time'].min() * 0.8)
    
    # Plot 4: BAR PLOT - mae vs Number of Molecules with Time Steps as grouped bars
    molecules_sorted = sorted(df['molecules'].unique())
    time_steps_sorted = sorted(df['time_step'].unique())
    
    # Set up bar positions
    n_molecules = len(molecules_sorted)
    n_time_steps = len(time_steps_sorted)
    bar_width = 0.8 / n_time_steps
    x_pos = np.arange(n_molecules)
    
    # Colors for different time steps
    colors_bar = plt.cm.Set3(np.linspace(0, 1, n_time_steps))
    
    # Create bars for each time step
    for i, time_step in enumerate(time_steps_sorted):
        means = []
        stds = []
        
        for molecules in molecules_sorted:
            data_point = df[(df['molecules'] == molecules) & (df['time_step'] == time_step)]
            if not data_point.empty:
                means.append(data_point['mae'].iloc[0])
                stds.append(data_point['std_mae'].iloc[0])
            else:
                means.append(0)
                stds.append(0)
        print('time_step', time_step, 'stds', stds)
        # Plot bars with error bars
        bar_positions = x_pos + i * bar_width - (n_time_steps - 1) * bar_width / 2
        bars = ax4.bar(bar_positions, means, bar_width, 
                      yerr=stds, capsize=3,
                      color=colors_bar[i], alpha=0.8,
                      label=f'dt = {time_step}',
                      edgecolor='black', linewidth=0.5)
    
    # Customize the bar plot
    ax4.set_xlabel('Number of Molecules', fontsize=17)
    ax4.set_ylabel('Mean mae (%)', fontsize=17)
    ax4.set_title('mae vs Number of Molecules by Time Step', fontsize=17)
    
    # Set x-axis labels
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{int(mol):.0e}' for mol in molecules_sorted], fontsize=12)
    
    # Add legend
    ax4.legend(ncol=4, fontsize=10, loc='upper right')
    ax4.grid(axis='y',which="both", alpha=0.8)
    
    # Set y-axis to start from 0 for better bar plot visualization
    ax4.set_ylim(bottom=1.0)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"analysis_plots_from_csv_{os.path.basename(csv_filepath).replace('.csv', '.png')}"
    plot_path = os.path.join(base_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()  # Display the plot
    print(f"Analysis plots saved to: {plot_path}")




    # === NEW PLOT: Standard Deviation vs Number of Molecules ===
    fig_std, ax_std = plt.subplots(figsize=(8, 6))
    
    for i, ts in enumerate(sorted(time_steps)):
        if ts <= 0.002 and ts >0.0001: 
            data = df[df['time_step'] == ts].sort_values('molecules')
            ax_std.loglog(data['molecules'], data['std_mae'], 'o-', 
                        color=colors[i], linewidth=2, markersize=6, 
                        label=f'dt={ts}')
        
    ax_std.set_xlabel('Number of Molecules', fontsize=15)
    ax_std.set_ylabel('Standard Deviation of MAE', fontsize=15)
    ax_std.set_title('Standard Deviation of MAE vs Number of Molecules', fontsize=16)
    ax_std.legend(ncol=2, fontsize=10)
    ax_std.grid(True, which="both", alpha=0.8)
    ax_std.set_xlim(left=df['molecules'].min() * 0.8)
    ax_std.set_ylim(bottom=df['std_mae'].min() * 0.8,
                    top=df['std_mae'].max() * 1.3)
    
    # Save the std plot
    std_plot_filename = f"std_vs_molecules_{os.path.basename(csv_filepath).replace('.csv', '.png')}"
    std_plot_path = os.path.join(base_dir, std_plot_filename)
    fig_std.savefig(std_plot_path, dpi=300, bbox_inches='tight')
    print(f"Standard deviation plot saved to: {std_plot_path}")

# Usage example:
if __name__ == "__main__":
    # Replace with your actual CSV file path
    csv_file_path = "/home/nguyt16@ds.vanderbilt.edu/HIPASim/hipa/tests/figs/single_cylinder_validation_2025-09-04_10-17-57/single_cylinder_validation_summary_a0.5_D02.0_threshold0.05.csv"
    
    # Load and plot
    df = load_and_plot_from_csv(csv_file_path)
    
    # # You can also do additional analysis
    # print("\nData summary:")
    # print(df.describe())